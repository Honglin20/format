# Architecture Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize `src/` into a four-layer dependency architecture (Math → Ops → Integration → Tools) with observer/ as cross-cutting infrastructure.

**Architecture:** Files move into cohesive packages organized by dependency layer. No class APIs change except the formats/quantize circular dependency fix (one low-level function relocation). All existing abstractions (FormatBase, QuantScheme, OpQuantConfig, etc.) are preserved.

**Tech Stack:** Python 3.x, PyTorch, pytest (1,416 passing baseline)

---

### Task 1: Break circular dependency formats ↔ quantize

**Files:**
- Create: `src/formats/_core.py`
- Modify: `src/quantize/elemwise.py:57-110`
- Modify: `src/formats/base.py:97`
- Modify: `src/quantize/bfloat_quantize.py:11`

**Step 1: Move `_quantize_elemwise_core` to formats/_core.py**

Cut the function `_quantize_elemwise_core` (lines 57-109) from `src/quantize/elemwise.py` and paste into new file `src/formats/_core.py`. The function is a pure format-level operation — it only needs bits/ebits/max_norm, not any quantize/ infrastructure.

New file `src/formats/_core.py`:

```python
"""Low-level format quantization primitives (shared by formats/ and quantize/)."""
import torch


def _round_mantissa(A, bits, round_mode, clamp=False):
    """Round mantissa according to round_mode."""
    if round_mode == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round_mode == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round_mode == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round_mode == "even":
        absA = torch.abs(A)
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise ValueError(f"Unrecognized round_mode {round_mode!r}")
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def _elemwise_core(A, bits, exp_bits, max_norm, round_mode='nearest',
                   saturate_normals=False, allow_denorm=True):
    """Element-wise quantization to a given number representation.

    Pure function: no dependency on FormatBase, QuantScheme, or any quantize/ module.
    """
    from src.formats.base import compute_min_norm

    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")
        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    if not allow_denorm and exp_bits > 0:
        min_norm = compute_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))
        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    out = _safe_lshift(out, bits - 2, private_exp)
    out = _round_mantissa(out, bits, round_mode, clamp=False)
    out = _safe_rshift(out, bits - 2, private_exp)

    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                          torch.sign(out) * float("Inf"), out)

    out[A == float("Inf")] = float("Inf")
    out[A == -float("Inf")] = -float("Inf")
    out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        out = torch.sparse_coo_tensor(sparse_A.indices(), out,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)
        return out

    return out
```

**Step 2: Remove `_quantize_elemwise_core` and helpers from quantize/elemwise.py**

Remove `_safe_lshift`, `_safe_rshift`, `_round_mantissa`, and `_quantize_elemwise_core` from `src/quantize/elemwise.py`. Keep `_quantize_elemwise`, `_quantize_bfloat`, `_quantize_fp`, and `quantize`.

**Step 3: Update imports in formats/base.py**

Change line 97 from:
```python
from src.quantize.elemwise import _quantize_elemwise_core
```
to:
```python
from src.formats._core import _elemwise_core as _quantize_elemwise_core
```
(Using alias preserves the method body without changes.)

**Step 4: Update quantize/elemwise.py to re-export from formats/_core**

Add import at top of `quantize/elemwise.py`:
```python
from src.formats._core import _elemwise_core as _quantize_elemwise_core, _round_mantissa, _safe_lshift, _safe_rshift
```

This preserves the public names for any internal users.

**Step 5: Update quantize/bfloat_quantize.py**

If `bfloat_quantize.py` imports `_quantize_elemwise_core` directly, update the import path.

**Step 6: Run tests**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```
Expected: 1,416 passed (no regression).

**Step 7: Commit**

```bash
git add src/formats/_core.py src/quantize/elemwise.py src/formats/base.py src/quantize/bfloat_quantize.py
git commit -m "fix: break circular dependency between formats and quantize

Move _quantize_elemwise_core to formats/_core.py as _elemwise_core.
This pure function only needs bits/ebits/max_norm — no dependency on
any quantize/ module. Both formats/ and quantize/ import from formats/_core."
```

---

### Task 2: Create observer/ package

**Files:**
- Create: `src/observer/__init__.py`
- Move: `src/analysis/events.py` → `src/observer/events.py`
- Move: `src/analysis/mixin.py` → `src/observer/mixin.py`
- Move: `src/analysis/observer.py` → `src/observer/observer.py`
- Modify: `src/ops/linear.py:19`, `conv.py:19`, `norm.py:19`, `activations.py:16`, `pooling.py:12`, `softmax.py:10`
- Modify: `src/context/_patches.py:59`
- Modify: `src/analysis/__init__.py`
- Modify: Test files with observer imports

**Step 1: Create observer/ package directory and __init__.py**

```bash
mkdir -p src/observer
```

Write `src/observer/__init__.py`:
```python
"""Cross-cutting observer infrastructure for quantized operator event collection."""

from .events import QuantEvent
from .mixin import ObservableMixin
from .observer import ObserverBase, SliceAwareObserver, SliceKey

__all__ = [
    "QuantEvent",
    "ObservableMixin",
    "ObserverBase",
    "SliceAwareObserver",
    "SliceKey",
]
```

**Step 2: Move files**

```bash
git mv src/analysis/events.py src/observer/events.py
git mv src/analysis/mixin.py src/observer/mixin.py
git mv src/analysis/observer.py src/observer/observer.py
```

**Step 3: Fix internal imports in moved files**

In `src/observer/mixin.py`, change line 13:
```python
from src.analysis.events import QuantEvent
```
to:
```python
from src.observer.events import QuantEvent
```

In `src/observer/observer.py`, change line 22:
```python
from src.analysis.events import QuantEvent
```
to:
```python
from src.observer.events import QuantEvent
```

**Step 4: Update ops/ imports (6 files)**

In each of these files, change:
```python
from src.analysis.mixin import ObservableMixin
```
to:
```python
from src.observer.mixin import ObservableMixin
```

Files: `src/ops/linear.py:19`, `conv.py:19`, `norm.py:19`, `activations.py:16`, `pooling.py:12`, `softmax.py:10`

**Step 5: Update context/_patches.py**

Change line 59:
```python
from src.analysis.events import QuantEvent
```
to:
```python
from src.observer.events import QuantEvent
```

**Step 6: Update analysis/__init__.py to re-export from observer/**

Change `src/analysis/__init__.py`:
```python
from .events import QuantEvent         # old
from .mixin import ObservableMixin      # old
from .observer import ObserverBase, SliceAwareObserver, SliceKey  # old
```
to:
```python
from src.observer import QuantEvent, ObservableMixin, ObserverBase, SliceAwareObserver, SliceKey
```

Also add SliceKey to __all__ if not already there.

**Step 7: Run tests**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```
Expected: 1,416 passed.

**Step 8: Commit**

```bash
git add src/observer/ src/analysis/__init__.py src/ops/linear.py src/ops/conv.py src/ops/norm.py src/ops/activations.py src/ops/pooling.py src/ops/softmax.py src/context/_patches.py
git commit -m "refactor: extract observer/ cross-cutting package from analysis/

Move QuantEvent, ObservableMixin, ObserverBase, SliceAwareObserver to
observer/ package. ops/ and context/_patches now import from observer/
instead of analysis/, fixing the reverse dependency where core ops
depended on the analysis tool layer."
```

---

### Task 3: Merge small files

**Part A: Merge cost/defaults.py into cost/device.py**

**Files:** `src/cost/defaults.py`, `src/cost/device.py`, `src/cost/__init__.py`

Move all constants from `defaults.py` into `device.py` (above the dataclass). Update `device.py` to use the constants directly instead of `from . import defaults`. Delete `defaults.py`.

Update `cost/__init__.py` if it re-exports anything from `defaults`.

```bash
# After code changes:
pytest src/tests/test_cost_*.py -q
```

**Part B: Merge pipeline/protocol.py into pipeline/runner.py**

**Files:** `src/pipeline/protocol.py`, `src/pipeline/runner.py`

Move the `EvalFn` Protocol class (15 lines) to the top of `runner.py`. Delete `protocol.py`.

Update `pipeline/__init__.py` if it re-exports from `protocol`.

```bash
pytest src/tests/test_pipeline_*.py -q
```

**Part C: Merge viz/save.py into viz/figures.py**

**Files:** `src/viz/save.py`, `src/viz/figures.py`

Move `save_figure()` into `figures.py` as a module-level function. Delete `save.py`.

Update `viz/__init__.py` if it re-exports from `save`.

```bash
pytest src/tests/test_viz_*.py -q
```

**Step: Commit**

```bash
git add src/cost/ src/pipeline/ src/viz/
git commit -m "refactor: merge files below 30 lines into parent modules

- cost/defaults.py → cost/device.py (23+22 = 45 lines)
- pipeline/protocol.py → pipeline/runner.py (15 lines)
- viz/save.py → viz/figures.py (21 lines)

Each merged file contained a single tightly-coupled concern."
```

---

### Task 4: Create session/ package (Integration layer)

**Files:**
- Create: `src/session/__init__.py`
- Create: `src/session/_context.py` (merge of _state + _stack + quantize_context)
- Move: `src/context/_patches.py` → `src/session/_patches.py`
- Move: `src/mapping/quantize_model.py` → `src/session/_model.py`
- Move: `src/session.py` → `src/session/_session.py`
- Modify: ALL files that import from `src.session`, `src.mapping`, `src.context`

**Step 1: Create session/ directory**

```bash
mkdir -p src/session
```

**Step 2: Merge context files into session/_context.py**

Combine `context/_state.py` (26 lines) + `context/_stack.py` (53 lines) + `context/quantize_context.py` (118 lines) into a single `src/session/_context.py` (~197 lines).

Structure:
```python
"""QuantizeContext: context manager for inline-op quantization interception."""
import contextvars
from dataclasses import dataclass, field
from typing import ...

from src.scheme.op_config import OpQuantConfig

# ── State ──
_EMPTY_CFG = OpQuantConfig()

@dataclass
class _CtxState: ...

_ctx_state: contextvars.ContextVar[...] = ...

# ── Stack ──
_module_stack: contextvars.ContextVar[...] = ...

def get_layer_name() -> str: ...
def install_stack_hooks(model) -> List: ...
def remove_stack_hooks(handles) -> None: ...

# ── QuantizeContext ──
class QuantizeContext: ...
```

Update internal imports: `from ._state import` / `from ._stack import` → direct references within the same file.

**Step 3: Move _patches.py**

```bash
git mv src/context/_patches.py src/session/_patches.py
```

Update imports in `_patches.py`:
- `from ._state import _ctx_state, _EMPTY_CFG` → `from src.session._context import _ctx_state, _EMPTY_CFG`
- `from ._stack import get_layer_name` → `from src.session._context import get_layer_name`
- `from src.observer.events import QuantEvent` (already updated in Task 2)

**Step 4: Move quantize_model.py**

```bash
git mv src/mapping/quantize_model.py src/session/_model.py
```

Update imports in `_model.py`:
- `from src.context.quantize_context import QuantizeContext` → `from src.session._context import QuantizeContext`
- All other imports stay the same (they import from `src.ops.*` and `src.scheme.*`)

**Step 5: Move session.py**

```bash
git mv src/session.py src/session/_session.py
```

Update imports in `_session.py`:
- `from src.mapping.quantize_model import quantize_model` → `from src.session._model import quantize_model`

All other imports (calibration, analysis, cost, scheme, transform, observer) stay the same.

**Step 6: Create session/__init__.py**

```python
"""Integration layer: model quantization lifecycle."""
from ._context import QuantizeContext, install_stack_hooks, remove_stack_hooks
from ._model import quantize_model
from ._session import QuantSession

__all__ = [
    "QuantSession",
    "QuantizeContext",
    "quantize_model",
    "install_stack_hooks",
    "remove_stack_hooks",
]
```

**Step 7: Update ALL imports across the codebase**

This is the largest step. Every file that imports from `src.session`, `src.mapping`, or `src.context` must be updated.

Affected files (identified by grep):

| Old import | New import |
|-----------|-----------|
| `from src.session import QuantSession` | `from src.session import QuantSession` (unchanged if using package) |
| `from src.mapping.quantize_model import quantize_model` | `from src.session import quantize_model` |
| `from src.context.quantize_context import QuantizeContext` | `from src.session import QuantizeContext` |
| `from src.context._state import ...` | `from src.session._context import ...` |
| `from src.context._stack import ...` | `from src.session._context import ...` |

Key files to update:
- `src/pipeline/runner.py` — imports from session
- `src/pipeline/format_study.py` — imports from session
- `src/analysis/e2e.py` — may import from session
- `src/analysis/context.py` — imports ObservableMixin
- `src/cost/report.py` — may import from session
- ALL test files that import from mapping/, context/, session

Use find-and-replace across the codebase:
```bash
# Find all files referencing old paths
grep -rl "from src.mapping" src/ tests/
grep -rl "from src.context" src/ tests/
grep -rl "import src.session" src/ tests/
```

**Step 8: Run tests**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```
Expected: 1,416 passed.

**Step 9: Commit**

```bash
git add src/session/ src/pipeline/ src/analysis/ src/cost/ src/tests/
git rm src/session.py src/mapping/quantize_model.py src/context/__init__.py src/context/_patches.py src/context/_state.py src/context/_stack.py src/context/quantize_context.py
git commit -m "refactor: create session/ integration layer package

Absorb session.py, mapping/quantize_model, and context/ into session/:
- _session.py (QuantSession — lifecycle orchestrator)
- _model.py (quantize_model — module→op mapping)
- _context.py (QuantizeContext + _CtxState + module stack)
- _patches.py (torch function interception)

Merge context/_state.py + _stack.py + quantize_context.py → _context.py.
Remove mapping/ and context/ packages."
```

---

### Task 5: Delete dead code

**Step 1: Delete config/**

```bash
git rm -r src/config/
```

No imports to update — zero references across the codebase.

**Step 2: Delete analysis/export.py**

```bash
git rm src/analysis/export.py
```

No imports to update — zero references, not re-exported from analysis/__init__.py.

**Step 3: Run tests**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```

**Step 4: Commit**

```bash
git commit -m "chore: delete dead code - config/ package and analysis/export.py

config/: empty __init__.py (0 bytes), zero imports across codebase.
analysis/export.py: 5-line docstring stub, zero executable code."
```

---

### Task 6: Create _utils/ package

**Step 1: Create _utils/ with appropriate contents**

```bash
mkdir -p src/_utils
```

Write `src/_utils/__init__.py`:
```python
"""Private internal utilities. Not part of the public API.

The leading underscore warns: `from src._utils import X` is deliberate
and you should know what you're doing.
"""
```

Move `src/analysis/slicing.py` → `src/_utils/slicing.py` (pure tensor slicing utility, not analysis-specific).

**Step 2: Update imports**

```bash
grep -rn "from src.analysis.slicing import" src/ tests/
```

Update each reference.

**Step 3: Run tests and commit**

```bash
pytest src/tests/test_slicing.py -q
git add src/_utils/
git commit -m "refactor: create _utils/ for private internal utilities"
```

---

### Task 7: Update CURRENT.md and final verification

**Step 1: Update docs/status/CURRENT.md**

Add a note about the completed refactoring.

**Step 2: Run full test suite**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```
Expected: 1,416 passed.

**Step 3: Verify no stale imports**

```bash
grep -rn "from src\.mapping" src/ tests/ && echo "STALE: mapping references remain" || echo "OK: no mapping references"
grep -rn "from src\.context" src/ tests/ && echo "STALE: context references remain" || echo "OK: no context references"
grep -rn "from src\.config" src/ tests/ && echo "STALE: config references remain" || echo "OK: no config references"
grep -rn "from src\.session import" src/ tests/ && echo "WARNING: direct session.py imports remain" || echo "OK: no old session imports"
```

**Step 4: Verify the dependency graph**

Confirm no reverse dependencies:
- `ops/` does NOT import `analysis/` (only `observer/`)
- `formats/` does NOT import `quantize/` (only via `_core.py`)
- `scheme/` does NOT import any other `src.*` (still a leaf)

**Step 5: Final commit**

```bash
git add docs/status/CURRENT.md
git commit -m "docs: update CURRENT.md after architecture refactor

Completed four-layer architecture refactor:
Math (formats transform scheme quantize) → Ops → Integration (session) → Tools.
Added observer/ cross-cutting package. Deleted config/, mapping/, context/."
```
