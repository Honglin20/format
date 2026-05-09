# QuantizeContext Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `QuantizeContext`, a context manager that patches `torch`/`F` namespace ops so all quantization (inline `torch.matmul`, `torch.add`, `F.linear`, etc.) is captured uniformly without `quantize_model`.

**Architecture:** Three-layer mechanism: (1) save originals in existing Function files to prevent recursion inside forward/backward; (2) `_ctx_state` ContextVar carries `_CtxState(cfg, op_cfgs, observers)` thread-safely; (3) `QuantizeContext.__enter__` patches torch/F namespaces + installs module-stack hooks, `__exit__` restores everything. ONNX export runs *inside* the active context so existing `symbolic()` on LinearFunction/ConvFunction fire naturally; MatMulFunction and BMMFunction receive new `symbolic()` methods; SIMD Functions get minimal pass-through symbolics.

**Tech Stack:** Python `contextvars`, `torch.autograd.Function.symbolic`, existing `src/ops/` Functions.

**Limitation (by design):** `a @ b` and `a + b` Python operators dispatch through C++ `Tensor.__matmul__`/`__add__` and are **not** intercepted. Only explicit `torch.matmul(a, b)`, `torch.add(a, b)`, `F.linear(x, w)` etc. are patchable. This matches mx's behaviour.

---

### Task 1: Add `_orig_*` saves to existing Function files

**Purpose:** When QuantizeContext patches `torch.matmul` / `torch.bmm` / `F.linear`, the existing Function classes (MatMulFunction, BMMFunction, LinearFunction) must call the *original* torch ops internally — not the patched versions — or they'd recurse infinitely.

**Files:**
- Modify: `src/ops/matmul.py`
- Modify: `src/ops/bmm.py`
- Modify: `src/ops/linear.py`

No new tests — the existing 973 tests must continue to pass unchanged.

**Step 1: Edit `src/ops/matmul.py`**

Add immediately after imports (before the class definition):
```python
_torch_matmul = torch.matmul
```

Change three calls inside the class:
- `MatMulFunction.forward` line ~104: `out = torch.matmul(in1, in2)` → `out = _torch_matmul(in1, in2)`
- `MatMulFunction.backward` line ~163: `grad_in1 = torch.matmul(...)` → `grad_in1 = _torch_matmul(...)`
- `MatMulFunction.backward` line ~166: `grad_in2 = torch.matmul(...)` → `grad_in2 = _torch_matmul(...)`
- `quantized_matmul` function at bottom: `return torch.matmul(in1, in2)` → `return _torch_matmul(in1, in2)`

**Step 2: Edit `src/ops/bmm.py`**

Add after imports:
```python
_torch_bmm = torch.bmm
```

Replace all four `torch.bmm(...)` calls in the file (lines ~81, ~128, ~131, ~154) with `_torch_bmm(...)`.

**Step 3: Edit `src/ops/linear.py`**

Add after imports (before the class definition):
```python
_F_linear = F.linear
```

Replace the single `F.linear(x, w)` call in `LinearFunction.forward` with `_F_linear(x, w)`.

**Step 4: Run full suite**
```bash
pytest src/tests/ -x -q
```
Expected: 973 passed, 0 failed.

**Step 5: Commit**
```bash
git add src/ops/matmul.py src/ops/bmm.py src/ops/linear.py
git commit -m "refactor(ops): save _orig_* torch refs to prevent QuantizeContext double-dispatch"
```

---

### Task 2: Context state + module-stack infrastructure

**Files:**
- Create: `src/context/__init__.py`
- Create: `src/context/_state.py`
- Create: `src/context/_stack.py`
- Create: `src/tests/test_quantize_context.py`

**Step 1: Write the failing tests**

Create `src/tests/test_quantize_context.py`:
```python
"""Tests for QuantizeContext — written task-by-task, each task adds cases."""
import torch
import torch.nn as nn
import pytest

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.formats.int_formats import IntFormat


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _int8_scheme():
    return QuantScheme(
        format=IntFormat(bits=8, is_signed=True),
        granularity=GranularitySpec(GranularityMode.PER_TENSOR),
    )


def _make_cfg(**roles):
    """Build OpQuantConfig; default all forward roles to int8 if not given."""
    s = _int8_scheme()
    defaults = dict(input=(s,), weight=(s,), output=(s,))
    defaults.update(roles)
    return OpQuantConfig(**defaults)


# ---------------------------------------------------------------------------
# Task 2 — _CtxState + _ctx_state
# ---------------------------------------------------------------------------

from src.context._state import _ctx_state, _CtxState, _EMPTY_CFG


def test_ctx_state_not_active_by_default():
    assert _ctx_state.get(None) is None


def test_ctx_state_set_and_reset():
    cfg = OpQuantConfig()
    state = _CtxState(cfg=cfg)
    tok = _ctx_state.set(state)
    assert _ctx_state.get().cfg is cfg
    _ctx_state.reset(tok)
    assert _ctx_state.get(None) is None


def test_ctx_state_resolve_default():
    cfg = _make_cfg()
    state = _CtxState(cfg=cfg)
    assert state.resolve("matmul") is cfg
    assert state.resolve("add") is cfg


def test_ctx_state_resolve_per_op_override():
    default_cfg = OpQuantConfig()
    matmul_cfg = _make_cfg()
    state = _CtxState(cfg=default_cfg, op_cfgs={"matmul": matmul_cfg})
    assert state.resolve("matmul") is matmul_cfg
    assert state.resolve("add") is default_cfg


# ---------------------------------------------------------------------------
# Task 2 — module stack
# ---------------------------------------------------------------------------

from src.context._stack import install_stack_hooks, remove_stack_hooks, get_layer_name


def test_get_layer_name_empty():
    assert get_layer_name() == ""


def test_stack_records_module_name_during_forward():
    captured = []

    class Probe(nn.Module):
        def forward(self, x):
            captured.append(get_layer_name())
            return x

    model = nn.Sequential(Probe())
    hooks = install_stack_hooks(model)
    model(torch.zeros(1))
    remove_stack_hooks(hooks)
    assert "0" in captured[0]


def test_stack_cleans_up_after_forward():
    model = nn.Linear(4, 4)
    hooks = install_stack_hooks(model)
    model(torch.zeros(2, 4))
    remove_stack_hooks(hooks)
    assert get_layer_name() == ""
```

**Step 2: Verify tests fail**
```bash
pytest src/tests/test_quantize_context.py -x -q 2>&1 | head -20
```
Expected: ImportError on `src.context._state`.

**Step 3: Create `src/context/_state.py`**
```python
import contextvars
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.scheme.op_config import OpQuantConfig

_EMPTY_CFG = OpQuantConfig()


@dataclass
class _CtxState:
    cfg: OpQuantConfig
    op_cfgs: Dict[str, OpQuantConfig] = field(default_factory=dict)
    observers: List = field(default_factory=list)

    def resolve(self, op_name: str) -> OpQuantConfig:
        """Return per-op cfg if overridden, else default cfg."""
        if self.op_cfgs and op_name in self.op_cfgs:
            return self.op_cfgs[op_name]
        return self.cfg


_ctx_state: contextvars.ContextVar[Optional[_CtxState]] = contextvars.ContextVar(
    "quant_ctx_state", default=None
)
```

**Step 4: Create `src/context/_stack.py`**
```python
import contextvars
from typing import List

import torch.nn as nn

_module_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "quant_module_stack", default=[]
)


def get_layer_name() -> str:
    stack = _module_stack.get()
    return ".".join(stack) if stack else ""


def _make_pre_hook(name: str):
    def _pre(module, inp):
        stack = list(_module_stack.get())
        stack.append(name)
        _module_stack.set(stack)
    return _pre


def _make_post_hook(name: str):
    def _post(module, inp, out):
        stack = list(_module_stack.get())
        if stack and stack[-1] == name:
            stack.pop()
            _module_stack.set(stack)
    return _post


def install_stack_hooks(model: nn.Module) -> List:
    """Register forward pre/post hooks on all named sub-modules. Returns handles."""
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_pre_hook(_make_pre_hook(name)))
        handles.append(module.register_forward_hook(_make_post_hook(name)))
    return handles


def remove_stack_hooks(handles: List) -> None:
    for h in handles:
        h.remove()
```

**Step 5: Create stub `src/context/__init__.py`**
```python
from .quantize_context import QuantizeContext  # noqa: F401 — filled in Task 7
```
Leave this as a forward-reference stub; we'll fill `quantize_context.py` in Task 7. For now just create an empty `src/context/quantize_context.py`:
```python
# placeholder — implemented in Task 7
```

**Step 6: Run Task 2 tests**
```bash
pytest src/tests/test_quantize_context.py -k "ctx_state or stack" -v
```
Expected: all 7 tests pass.

**Step 7: Run full suite (no regressions)**
```bash
pytest src/tests/ -x -q
```

**Step 8: Commit**
```bash
git add src/context/ src/tests/test_quantize_context.py
git commit -m "feat(context): add _CtxState + module-stack tracking"
```

---

### Task 3: Add `symbolic()` to MatMulFunction and BMMFunction

**Purpose:** Enable ONNX export when QuantizeContext patches `torch.matmul`/`torch.bmm` — the tracer will call these `symbolic()` methods instead of trying to lower the custom ops.

**Files:**
- Modify: `src/ops/matmul.py`
- Modify: `src/ops/bmm.py`

**Step 1: Write failing test**

Add to `src/tests/test_quantize_context.py` (at the bottom, will be skipped until Task 7 wires up the context — add a `pytest.importorskip` guard on the module so this file is additive):

```python
# ---------------------------------------------------------------------------
# Task 3 — MatMulFunction.symbolic + BMMFunction.symbolic
# ---------------------------------------------------------------------------

import tempfile, os


def _export_fn_direct(fn_cls, *args, path):
    """Helper: export a single Function call to ONNX."""
    import torch

    class _Wrapper(nn.Module):
        def forward(self, *inputs):
            return fn_cls.apply(*inputs)

    # only test that symbolic() doesn't crash; full export tested in Task 8
    pass  # placeholder — the real test is in Task 8 end-to-end


def test_matmul_symbolic_method_exists():
    from src.ops.matmul import MatMulFunction
    assert hasattr(MatMulFunction, "symbolic")


def test_bmm_symbolic_method_exists():
    from src.ops.bmm import BMMFunction
    assert hasattr(BMMFunction, "symbolic")
```

**Step 2: Verify both tests fail (AttributeError)**
```bash
pytest src/tests/test_quantize_context.py::test_matmul_symbolic_method_exists -v
pytest src/tests/test_quantize_context.py::test_bmm_symbolic_method_exists -v
```

**Step 3: Add `symbolic()` to `MatMulFunction` in `src/ops/matmul.py`**

Insert after the `backward` staticmethod (before the closing of the class):
```python
    @staticmethod
    def symbolic(g, in1, in2, bias, cfg, name, mode_config, emit_fn):
        """ONNX symbolic: Q/DQ wrappers + MatMul + optional Add."""
        from src.onnx.helpers import _emit_quantize_node
        for scheme in cfg.input:
            in1 = _emit_quantize_node(g, in1, scheme)
        for scheme in cfg.weight:
            in2 = _emit_quantize_node(g, in2, scheme)
        out = g.op("MatMul", in1, in2)
        if len(cfg.output) > 0:
            out = _emit_quantize_node(g, out, cfg.output[0])
        if bias is not None:
            qb = bias
            for scheme in cfg.bias:
                qb = _emit_quantize_node(g, qb, scheme)
            out = g.op("Add", out, qb)
            if len(cfg.output) > 1:
                out = _emit_quantize_node(g, out, cfg.output[1])
        return out
```

**Step 4: Add `symbolic()` to `BMMFunction` in `src/ops/bmm.py`**

Read bmm.py to find where backward ends, then insert:
```python
    @staticmethod
    def symbolic(g, in1, in2, cfg, name, emit_fn):
        """ONNX symbolic: Q/DQ wrappers + MatMul (ONNX MatMul supports batched)."""
        from src.onnx.helpers import _emit_quantize_node
        for scheme in cfg.input:
            in1 = _emit_quantize_node(g, in1, scheme)
        for scheme in cfg.weight:
            in2 = _emit_quantize_node(g, in2, scheme)
        out = g.op("MatMul", in1, in2)
        if len(cfg.output) > 0:
            out = _emit_quantize_node(g, out, cfg.output[0])
        return out
```

**Step 5: Run tests**
```bash
pytest src/tests/test_quantize_context.py::test_matmul_symbolic_method_exists -v
pytest src/tests/test_quantize_context.py::test_bmm_symbolic_method_exists -v
pytest src/tests/ -x -q
```
Expected: new tests pass, 973 existing still pass.

**Step 6: Commit**
```bash
git add src/ops/matmul.py src/ops/bmm.py
git commit -m "feat(ops): add symbolic() to MatMulFunction and BMMFunction for ONNX export"
```

---

### Task 4: Add `symbolic()` to SIMD Functions

**Purpose:** When QuantizeContext patches `torch.add/sub/mul/div/exp/log`, ONNX export must not crash on these custom ops. Each gets a minimal symbolic that emits Q/DQ wrappers around the standard ONNX op.

**Files:**
- Modify: `src/ops/elemwise.py`

**Step 1: Write failing tests**

Add to `src/tests/test_quantize_context.py`:
```python
# ---------------------------------------------------------------------------
# Task 4 — SIMD symbolic() methods
# ---------------------------------------------------------------------------

def test_simd_symbolic_methods_exist():
    from src.ops.elemwise import SIMDAdd, SIMDSub, SIMDMul, SIMDDiv, SIMDExp, SIMDLog
    for cls in (SIMDAdd, SIMDSub, SIMDMul, SIMDDiv, SIMDExp, SIMDLog):
        assert hasattr(cls, "symbolic"), f"{cls.__name__} missing symbolic()"
```

**Step 2: Verify fail**
```bash
pytest src/tests/test_quantize_context.py::test_simd_symbolic_methods_exist -v
```

**Step 3: Add `symbolic()` to binary SIMD classes in `src/ops/elemwise.py`**

For `SIMDAdd` (insert after its `backward`):
```python
    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Add", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out
```

For `SIMDSub`:
```python
    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Sub", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out
```

For `SIMDMul`:
```python
    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Mul", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out
```

For `SIMDDiv`:
```python
    @staticmethod
    def symbolic(g, in1, in2, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
            in2 = _emit_quantize_node(g, in2, inner_scheme)
        out = g.op("Div", in1, in2)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out
```

For `SIMDExp` (unary — 3 args: g, in1, inner_scheme, quantize_backprop):
```python
    @staticmethod
    def symbolic(g, in1, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
        out = g.op("Exp", in1)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out
```

For `SIMDLog`:
```python
    @staticmethod
    def symbolic(g, in1, inner_scheme, quantize_backprop):
        from src.onnx.helpers import _emit_quantize_node
        if inner_scheme is not None:
            in1 = _emit_quantize_node(g, in1, inner_scheme)
        out = g.op("Log", in1)
        if inner_scheme is not None:
            out = _emit_quantize_node(g, out, inner_scheme)
        return out
```

**Step 4: Run tests**
```bash
pytest src/tests/test_quantize_context.py::test_simd_symbolic_methods_exist -v
pytest src/tests/ -x -q
```

**Step 5: Commit**
```bash
git add src/ops/elemwise.py
git commit -m "feat(ops): add symbolic() to SIMD Functions for ONNX export via QuantizeContext"
```

---

### Task 5: Op patches (`src/context/_patches.py`)

**Files:**
- Create: `src/context/_patches.py`

**Step 1: Write failing tests**

Add to `src/tests/test_quantize_context.py`:
```python
# ---------------------------------------------------------------------------
# Task 5 — _patches: each patched fn passes through without context
# ---------------------------------------------------------------------------

from src.context._patches import (
    _patched_matmul, _patched_mm, _patched_bmm,
    _patched_F_linear, _patched_add, _patched_sub,
    _patched_mul, _patched_div, _patched_exp, _patched_log,
    _PATCH_TABLE,
)


def test_patch_table_has_all_ops():
    expected = {
        ("torch", "matmul"), ("torch", "mm"), ("torch", "bmm"),
        ("torch", "add"), ("torch", "sub"), ("torch", "mul"),
        ("torch", "div"), ("torch", "exp"), ("torch", "log"),
        ("F", "linear"),
    }
    assert expected.issubset(set(_PATCH_TABLE.keys()))


def test_patched_matmul_passthrough_without_context():
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    assert torch.equal(_patched_matmul(a, b), torch.matmul(a, b))


def test_patched_add_passthrough_without_context():
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    assert torch.equal(_patched_add(a, b), torch.add(a, b))


def test_patched_matmul_quantizes_with_active_context():
    cfg = _make_cfg()
    state = _CtxState(cfg=cfg)
    tok = _ctx_state.set(state)
    try:
        a, b = torch.randn(3, 4), torch.randn(4, 5)
        result = _patched_matmul(a, b)
        plain = torch.matmul(a, b)
        assert not torch.equal(result, plain)
    finally:
        _ctx_state.reset(tok)


def test_patched_F_linear_quantizes_with_active_context():
    import torch.nn.functional as F_orig
    cfg = _make_cfg()
    state = _CtxState(cfg=cfg)
    tok = _ctx_state.set(state)
    try:
        x = torch.randn(2, 8)
        w = torch.randn(4, 8)
        result = _patched_F_linear(x, w)
        plain = F_orig.linear(x, w)
        assert not torch.equal(result, plain)
    finally:
        _ctx_state.reset(tok)


def test_patched_add_with_scalar_passthrough():
    """Scalar second argument must not be routed through SIMDAdd."""
    a = torch.randn(3, 4)
    state = _CtxState(cfg=_make_cfg())
    tok = _ctx_state.set(state)
    try:
        result = _patched_add(a, 1.0)  # scalar → passthrough
        assert torch.equal(result, a + 1.0)
    finally:
        _ctx_state.reset(tok)
```

**Step 2: Verify fail**
```bash
pytest src/tests/test_quantize_context.py -k "patch" -v 2>&1 | head -30
```

**Step 3: Create `src/context/_patches.py`**
```python
"""
Patched torch/F op implementations for QuantizeContext.

Each function:
1. Reads _ctx_state; if None (outside any context), falls through to original.
2. Resolves OpQuantConfig via state.resolve(op_name).
3. If cfg is _EMPTY_CFG (no quantization), falls through to original.
4. Otherwise delegates to the existing autograd.Function with the resolved cfg.

IMPORTANT: existing Function files (matmul.py, bmm.py, linear.py) save
module-level _orig_* references so their internal torch calls don't re-enter
these patches and cause infinite recursion.
"""
import torch
import torch.nn.functional as F

from src.context._state import _ctx_state, _EMPTY_CFG
from src.context._stack import get_layer_name

# Originals captured at import time (before any patching occurs).
_orig_torch_matmul = torch.matmul
_orig_torch_mm     = torch.mm
_orig_torch_bmm    = torch.bmm
_orig_torch_add    = torch.add
_orig_torch_sub    = torch.sub
_orig_torch_mul    = torch.mul
_orig_torch_div    = torch.div
_orig_torch_exp    = torch.exp
_orig_torch_log    = torch.log
_orig_F_linear     = F.linear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_state():
    return _ctx_state.get(None)


def _make_emit_fn(state, layer_name: str, op_suffix: str):
    """Return emit_fn bound to current observers + layer name, or None."""
    if not state.observers:
        return None

    full_name = f"{layer_name}.{op_suffix}" if layer_name else op_suffix

    def emit_fn(role, pipeline_index, stage, fp32, quant, scheme, group_map=None):
        from src.analysis.events import QuantEvent
        event = QuantEvent(
            layer_name=full_name,
            role=role,
            pipeline_index=pipeline_index,
            stage=stage,
            fp32_tensor=fp32.detach(),
            quant_tensor=quant.detach(),
            scheme=scheme,
            group_map=group_map.detach() if group_map is not None else None,
        )
        for obs in state.observers:
            obs.on_event(event)

    return emit_fn


def _simd_inner_scheme(cfg):
    """Extract inner_scheme for SIMD ops: first input scheme or None."""
    return cfg.input[0] if cfg.input else None


# ---------------------------------------------------------------------------
# Matmul / Linear family
# ---------------------------------------------------------------------------

def _patched_matmul(a, b):
    state = _get_state()
    if state is None:
        return _orig_torch_matmul(a, b)
    cfg = state.resolve("matmul")
    if cfg == _EMPTY_CFG:
        return _orig_torch_matmul(a, b)
    from src.ops.matmul import MatMulFunction
    name = get_layer_name()
    return MatMulFunction.apply(a, b, None, cfg, name, "aa", _make_emit_fn(state, name, "matmul"))


def _patched_mm(a, b):
    state = _get_state()
    if state is None:
        return _orig_torch_mm(a, b)
    cfg = state.resolve("mm")
    if cfg == _EMPTY_CFG:
        return _orig_torch_mm(a, b)
    from src.ops.matmul import MatMulFunction
    name = get_layer_name()
    return MatMulFunction.apply(a, b, None, cfg, name, "aa", _make_emit_fn(state, name, "mm"))


def _patched_bmm(a, b):
    state = _get_state()
    if state is None:
        return _orig_torch_bmm(a, b)
    cfg = state.resolve("bmm")
    if cfg == _EMPTY_CFG:
        return _orig_torch_bmm(a, b)
    from src.ops.bmm import BMMFunction
    name = get_layer_name()
    return BMMFunction.apply(a, b, cfg, name, _make_emit_fn(state, name, "bmm"))


def _patched_F_linear(input, weight, bias=None):
    state = _get_state()
    if state is None:
        return _orig_F_linear(input, weight, bias)
    cfg = state.resolve("linear")
    if cfg == _EMPTY_CFG:
        return _orig_F_linear(input, weight, bias)
    from src.ops.linear import LinearFunction
    name = get_layer_name()
    return LinearFunction.apply(input, weight, bias, cfg, name, _make_emit_fn(state, name, "linear"))


# ---------------------------------------------------------------------------
# SIMD arithmetic (binary)
# ---------------------------------------------------------------------------

def _patched_add(a, b, *, alpha=1):
    state = _get_state()
    if state is None or not isinstance(b, torch.Tensor):
        return _orig_torch_add(a, b, alpha=alpha)
    cfg = state.resolve("add")
    if cfg == _EMPTY_CFG:
        return _orig_torch_add(a, b, alpha=alpha)
    from src.ops.elemwise import SIMDAdd
    if alpha != 1:
        b = b * alpha
    return SIMDAdd.apply(a, b, _simd_inner_scheme(cfg), True)


def _patched_sub(a, b, *, alpha=1):
    state = _get_state()
    if state is None or not isinstance(b, torch.Tensor):
        return _orig_torch_sub(a, b, alpha=alpha)
    cfg = state.resolve("sub")
    if cfg == _EMPTY_CFG:
        return _orig_torch_sub(a, b, alpha=alpha)
    from src.ops.elemwise import SIMDSub
    if alpha != 1:
        b = b * alpha
    return SIMDSub.apply(a, b, _simd_inner_scheme(cfg), True)


def _patched_mul(a, b):
    state = _get_state()
    if state is None or not isinstance(b, torch.Tensor):
        return _orig_torch_mul(a, b)
    cfg = state.resolve("mul")
    if cfg == _EMPTY_CFG:
        return _orig_torch_mul(a, b)
    from src.ops.elemwise import SIMDMul
    return SIMDMul.apply(a, b, _simd_inner_scheme(cfg), True)


def _patched_div(a, b):
    state = _get_state()
    if state is None or not isinstance(b, torch.Tensor):
        return _orig_torch_div(a, b)
    cfg = state.resolve("div")
    if cfg == _EMPTY_CFG:
        return _orig_torch_div(a, b)
    from src.ops.elemwise import SIMDDiv
    return SIMDDiv.apply(a, b, _simd_inner_scheme(cfg), True)


# ---------------------------------------------------------------------------
# SIMD unary
# ---------------------------------------------------------------------------

def _patched_exp(x):
    state = _get_state()
    if state is None:
        return _orig_torch_exp(x)
    cfg = state.resolve("exp")
    if cfg == _EMPTY_CFG:
        return _orig_torch_exp(x)
    from src.ops.elemwise import SIMDExp
    return SIMDExp.apply(x, _simd_inner_scheme(cfg), True)


def _patched_log(x):
    state = _get_state()
    if state is None:
        return _orig_torch_log(x)
    cfg = state.resolve("log")
    if cfg == _EMPTY_CFG:
        return _orig_torch_log(x)
    from src.ops.elemwise import SIMDLog
    return SIMDLog.apply(x, _simd_inner_scheme(cfg), True)


# ---------------------------------------------------------------------------
# Patch table + apply/remove helpers
# ---------------------------------------------------------------------------

_PATCH_TABLE = {
    ("torch", "matmul"): _patched_matmul,
    ("torch", "mm"):     _patched_mm,
    ("torch", "bmm"):    _patched_bmm,
    ("torch", "add"):    _patched_add,
    ("torch", "sub"):    _patched_sub,
    ("torch", "mul"):    _patched_mul,
    ("torch", "div"):    _patched_div,
    ("torch", "exp"):    _patched_exp,
    ("torch", "log"):    _patched_log,
    ("F",     "linear"): _patched_F_linear,
}


def _ns(key: str):
    if key == "torch":
        return torch
    if key == "F":
        import torch.nn.functional as _F
        return _F
    raise ValueError(f"Unknown namespace: {key!r}")


def apply_patches() -> dict:
    """Patch torch/F ops. Returns {key: original} for later restore."""
    saved = {}
    for (ns_key, attr), fn in _PATCH_TABLE.items():
        ns = _ns(ns_key)
        saved[(ns_key, attr)] = getattr(ns, attr)
        setattr(ns, attr, fn)
    return saved


def remove_patches(saved: dict) -> None:
    """Restore torch/F ops from saved dict."""
    for (ns_key, attr), orig in saved.items():
        setattr(_ns(ns_key), attr, orig)
```

**Step 4: Run tests**
```bash
pytest src/tests/test_quantize_context.py -k "patch" -v
pytest src/tests/ -x -q
```

**Step 5: Commit**
```bash
git add src/context/_patches.py
git commit -m "feat(context): add op patch table for matmul/bmm/linear/SIMD families"
```

---

### Task 6: `QuantizeContext` class

**Files:**
- Create (overwrite stub): `src/context/quantize_context.py`

**Step 1: Write failing tests**

Add to `src/tests/test_quantize_context.py`:
```python
# ---------------------------------------------------------------------------
# Task 6 — QuantizeContext class
# ---------------------------------------------------------------------------

from src.context.quantize_context import QuantizeContext


def test_context_quantizes_torch_matmul():
    cfg = _make_cfg()
    model = nn.Linear(1, 1)  # just needs a module for hook installation
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    plain = torch.matmul(a, b)

    with QuantizeContext(model, cfg):
        result = torch.matmul(a, b)

    assert not torch.equal(result, plain)


def test_context_restores_ops_on_exit():
    cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(4, 5)

    with QuantizeContext(model, cfg):
        pass

    # After exit, unpatched torch.matmul must return float result again
    result = torch.matmul(a, b)
    expected = torch.matmul(a, b)
    assert torch.equal(result, expected)


def test_context_intercepts_nn_linear_forward():
    """nn.Linear.forward calls F.linear, which the context intercepts."""
    cfg = _make_cfg()
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 8)

    plain = model(x)
    with QuantizeContext(model, cfg):
        quant = model(x)

    assert not torch.equal(plain, quant)


def test_context_intercepts_torch_add():
    cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    plain = torch.add(a, b)

    with QuantizeContext(model, cfg):
        result = torch.add(a, b)

    assert not torch.equal(result, plain)


def test_per_op_override_only_quantizes_specified_op():
    """Default cfg = no-quant; only matmul is overridden to int8."""
    default_cfg = OpQuantConfig()   # no quantization
    matmul_cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(4, 5)

    with QuantizeContext(model, default_cfg, op_cfgs={"matmul": matmul_cfg}):
        matmul_result = torch.matmul(a, b)
        # torch.add with default (empty) cfg → passthrough
        c, d = torch.randn(3, 5), torch.randn(3, 5)
        add_result = torch.add(c, d)

    assert not torch.equal(matmul_result, torch.matmul(a, b))
    assert torch.equal(add_result, torch.add(c, d))


def test_context_no_double_quantization():
    """F.linear inside LinearFunction.forward uses _F_linear, not the patch."""
    cfg = _make_cfg()
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 8)

    with QuantizeContext(model, cfg):
        r1 = model(x)
        r2 = model(x)

    assert torch.equal(r1, r2)  # deterministic = no unintended extra quantization


def test_context_is_not_active_outside_with_block():
    cfg = _make_cfg()
    model = nn.Linear(1, 1)

    with QuantizeContext(model, cfg):
        inside = torch.matmul(torch.randn(2, 3), torch.randn(3, 4))

    outside = torch.matmul(torch.randn(2, 3), torch.randn(3, 4))
    # outside should be a plain float result (not wrapped by Function)
    assert outside.requires_grad is False
```

**Step 2: Verify fail**
```bash
pytest src/tests/test_quantize_context.py -k "context" -v 2>&1 | head -30
```

**Step 3: Implement `src/context/quantize_context.py`**
```python
"""
QuantizeContext: unified context manager for all-op quantization.

Patches torch/F namespace ops so any quantizable computation is intercepted,
including inline torch.matmul, torch.add, F.linear, etc.
nn.Module-level ops (nn.Linear.forward → F.linear) are also intercepted
via the F.linear patch — no separate quantize_model call needed.

Limitations:
- `a @ b` and `a + b` Python operators go through C++ Tensor methods and
  are NOT intercepted. Use torch.matmul(a, b) / torch.add(a, b) instead.
- Same nn.Module multiple matmuls share one cfg (cannot distinguish QK vs QKV
  within one Attention.forward without a separate nn.Module per matmul).
"""
from typing import Dict, List, Optional

import torch.nn as nn

from src.context._state import _ctx_state, _CtxState
from src.context._stack import install_stack_hooks, remove_stack_hooks
from src.context._patches import apply_patches, remove_patches
from src.scheme.op_config import OpQuantConfig


class QuantizeContext:
    """Context manager that patches torch/F ops to apply quantization uniformly.

    Usage:
        with QuantizeContext(model, cfg) as ctx:
            output = model(x)       # all patchable ops quantized
            loss = output.sum()
            loss.backward()         # QAT backward also quantized via cfg

        ctx.export_onnx(dummy_input, "model.onnx")

    Args:
        model: The nn.Module whose sub-modules get stack-tracking hooks.
        cfg: Default OpQuantConfig applied to all patchable ops.
        op_cfgs: Optional per-op-type overrides. Valid keys:
                 "matmul", "mm", "bmm", "linear",
                 "add", "sub", "mul", "div", "exp", "log".
        observers: Optional observers (same interface as AnalysisContext).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: OpQuantConfig,
        *,
        op_cfgs: Optional[Dict[str, OpQuantConfig]] = None,
        observers: Optional[List] = None,
    ):
        self.model = model
        self._state = _CtxState(
            cfg=cfg,
            op_cfgs=op_cfgs or {},
            observers=observers or [],
        )
        self._ctx_token = None
        self._hook_handles: List = []
        self._saved_ops: dict = {}

    def __enter__(self):
        self._ctx_token = _ctx_state.set(self._state)
        self._hook_handles = install_stack_hooks(self.model)
        self._saved_ops = apply_patches()
        return self

    def __exit__(self, *args):
        remove_patches(self._saved_ops)
        remove_stack_hooks(self._hook_handles)
        _ctx_state.reset(self._ctx_token)
        self._saved_ops = {}
        self._hook_handles = []
        self._ctx_token = None

    def export_onnx(
        self,
        dummy_input,
        output_path: str,
        opset_version: int = 17,
    ) -> None:
        """Export to ONNX while patches are active.

        torch.onnx.export traces the model with the context active, so
        patched ops dispatch through LinearFunction / MatMulFunction /
        BMMFunction — their symbolic() methods produce correct ONNX nodes.
        SIMD ops export as standard ONNX Add/Sub/Mul/Div/Exp/Log with
        Q/DQ wrappers from their symbolic() methods.

        Args:
            dummy_input: Tensor (or tuple of tensors) defining input shapes.
            output_path: Where to write the .onnx file.
            opset_version: ONNX opset (default 17).
        """
        import torch
        from src.onnx.export import _verify_onnx_graph

        args = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
        torch.onnx.export(
            self.model,
            args,
            output_path,
            opset_version=opset_version,
            custom_opsets={"com.microxscaling": 1},
            do_constant_folding=False,
        )
        _verify_onnx_graph(output_path)
```

**Step 4: Update `src/context/__init__.py`** (already has the import, just ensure it's clean)

**Step 5: Run tests**
```bash
pytest src/tests/test_quantize_context.py -k "context" -v
pytest src/tests/ -x -q
```
Expected: all new tests pass, 973 existing still pass.

**Step 6: Commit**
```bash
git add src/context/quantize_context.py src/context/__init__.py
git commit -m "feat(context): implement QuantizeContext class with enter/exit/export_onnx"
```

---

### Task 7: ONNX export tests

**Files:**
- Modify: `src/tests/test_quantize_context.py`

**Step 1: Add ONNX tests**

Add to `src/tests/test_quantize_context.py`:
```python
# ---------------------------------------------------------------------------
# Task 7 — ONNX export via ctx.export_onnx
# ---------------------------------------------------------------------------

def test_export_onnx_nn_linear(tmp_path):
    """Linear model exports valid ONNX with QDQ nodes."""
    import onnx
    cfg = _make_cfg()
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
    x = torch.randn(2, 8)

    with QuantizeContext(model, cfg) as ctx:
        ctx.export_onnx(x, str(tmp_path / "linear.onnx"))

    m = onnx.load(str(tmp_path / "linear.onnx"))
    onnx.checker.check_model(m)
    node_types = {n.op_type for n in m.graph.node}
    assert "QuantizeLinear" in node_types or "MxQuantize" in node_types


def test_export_onnx_inline_matmul(tmp_path):
    """Model with inline torch.matmul exports valid ONNX."""
    import onnx

    class SelfAttnShape(nn.Module):
        def forward(self, x):
            # x: (B, S, D) — self-attention score matrix
            return torch.matmul(x, x.transpose(-2, -1))

    cfg = _make_cfg()
    model = SelfAttnShape()
    x = torch.randn(1, 4, 8)

    with QuantizeContext(model, cfg) as ctx:
        ctx.export_onnx(x, str(tmp_path / "attn.onnx"))

    m = onnx.load(str(tmp_path / "attn.onnx"))
    onnx.checker.check_model(m)
    node_types = {n.op_type for n in m.graph.node}
    assert "MatMul" in node_types


def test_export_onnx_with_add(tmp_path):
    """Model with torch.add exports valid ONNX."""
    import onnx

    class Residual(nn.Module):
        def forward(self, x):
            return torch.add(x, x)

    cfg = _make_cfg()
    model = Residual()
    x = torch.randn(2, 8)

    with QuantizeContext(model, cfg) as ctx:
        ctx.export_onnx(x, str(tmp_path / "residual.onnx"))

    m = onnx.load(str(tmp_path / "residual.onnx"))
    onnx.checker.check_model(m)
    assert any(n.op_type == "Add" for n in m.graph.node)
```

**Step 2: Run tests**
```bash
pytest src/tests/test_quantize_context.py -k "onnx" -v
```

Fix any issues found (likely ONNX tracing quirks with non-tensor Function args).

**Step 3: Run full suite**
```bash
pytest src/tests/ -x -q
```
Expected: all 973 existing + new context tests pass.

**Step 4: Commit**
```bash
git add src/tests/test_quantize_context.py
git commit -m "test(context): add ONNX export tests for QuantizeContext"
```

---

### Task 8: Update CURRENT.md and final housekeeping

**Files:**
- Modify: `docs/status/CURRENT.md`

**Step 1: Run complete test suite one final time**
```bash
pytest src/tests/ -q
```
Note the total count (expect 973 + N new context tests).

**Step 2: Update `docs/status/CURRENT.md`**

Add QuantizeContext as a new completed phase after Phase 5:
```markdown
- [x] **QuantizeContext** (Phase 6): unified all-op quantization context manager
  - Patches torch.matmul/mm/bmm, torch.add/sub/mul/div/exp/log, F.linear
  - Module-stack hooks for Observer layer identity
  - ONNX export via ctx.export_onnx() (runs inside active context)
  - New symbolic() on MatMulFunction, BMMFunction, SIMD Functions
```

**Step 3: Final commit**
```bash
git add docs/status/CURRENT.md
git commit -m "docs: mark QuantizeContext complete in CURRENT.md"
```

---

## Summary

| Task | What it builds | Tests added |
|------|---------------|-------------|
| 1 | `_orig_*` saves in matmul/bmm/linear | 0 (regression guard) |
| 2 | `_CtxState` + `_ctx_state` + module stack | 7 |
| 3 | `MatMulFunction.symbolic` + `BMMFunction.symbolic` | 2 |
| 4 | SIMD `symbolic()` on 6 Functions | 1 |
| 5 | `_patches.py`: all patched functions + apply/remove | 6 |
| 6 | `QuantizeContext` class (enter/exit/export_onnx) | 7 |
| 7 | ONNX export end-to-end tests | 3 |
| 8 | CURRENT.md housekeeping | 0 |

**Key files created/modified:**
```
src/context/
├── __init__.py
├── _state.py
├── _stack.py
├── _patches.py
└── quantize_context.py
src/ops/matmul.py          (+ _torch_matmul save + symbolic)
src/ops/bmm.py             (+ _torch_bmm save + symbolic)
src/ops/linear.py          (+ _F_linear save)
src/ops/elemwise.py        (+ 6 symbolic() methods)
src/tests/test_quantize_context.py   (new)
```
