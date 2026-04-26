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
