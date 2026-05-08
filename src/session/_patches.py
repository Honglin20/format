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
import functools
import torch
import torch.nn.functional as F

from src.quantize.elemwise import _is_in_quantize
from src.session._context import _ctx_state, _EMPTY_CFG
from src.session._context import get_layer_name

# Eagerly import Function classes so their module-level _torch_* saves are
# captured BEFORE apply_patches() is ever called. If these were deferred
# imports (inside function bodies), matmul.py etc. could be imported while
# patches are already active — causing _torch_matmul to capture _patched_matmul
# and creating infinite recursion on the very first forward call.
from src.ops.matmul import MatMulFunction
from src.ops.bmm import BMMFunction
from src.ops.linear import LinearFunction
from src.ops.elemwise import SIMDAdd, SIMDSub, SIMDMul, SIMDDiv, SIMDExp, SIMDLog

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

# Tensor dunder originals (operator overloading bypasses torch.add etc.)
_orig_tensor_add    = torch.Tensor.__add__
_orig_tensor_radd   = torch.Tensor.__radd__
_orig_tensor_sub    = torch.Tensor.__sub__
_orig_tensor_rsub   = torch.Tensor.__rsub__
_orig_tensor_mul    = torch.Tensor.__mul__
_orig_tensor_rmul   = torch.Tensor.__rmul__
_orig_tensor_truediv  = torch.Tensor.__truediv__
_orig_tensor_rtruediv = torch.Tensor.__rtruediv__


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
        from src.observer.events import QuantEvent
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
    """Extract inner_scheme for SIMD ops: storage (elemwise) first, then input.

    Returns a copy with IdentityTransform so user-configured transforms
    (SmoothQuant, Hadamard) are not spuriously applied to SIMD intermediates.
    """
    from src.scheme.transform import IdentityTransform
    scheme = cfg.storage or cfg.input
    if scheme is not None and type(scheme.transform) is not IdentityTransform:
        from src.scheme.quant_scheme import QuantScheme
        scheme = QuantScheme(
            format=scheme.format,
            granularity=scheme.granularity,
            transform=IdentityTransform(),
            round_mode=scheme.round_mode,
            scale_storage=scheme.scale_storage,
        )
    return scheme


def _patched(op_name, orig_fn):
    """Decorator factory: wraps a fn with state/cfg guard for patched torch ops.

    The decorated function signature is ``fn(state, cfg, *args, **kwargs)``.
    The guard (no active context, _EMPTY_CFG passthrough) is handled automatically.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            state = _get_state()
            if state is None:
                return orig_fn(*args, **kwargs)
            cfg = state.resolve(op_name)
            if cfg == _EMPTY_CFG:
                return orig_fn(*args, **kwargs)
            return fn(state, cfg, *args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Matmul / Linear family
# ---------------------------------------------------------------------------

@_patched("matmul", _orig_torch_matmul)
def _patched_matmul(state, cfg, a, b):
    name = get_layer_name()
    return MatMulFunction.apply(a, b, None, cfg, name, "aa", _make_emit_fn(state, name, "matmul"))


@_patched("mm", _orig_torch_mm)
def _patched_mm(state, cfg, a, b):
    name = get_layer_name()
    return MatMulFunction.apply(a, b, None, cfg, name, "aa", _make_emit_fn(state, name, "mm"))


@_patched("bmm", _orig_torch_bmm)
def _patched_bmm(state, cfg, a, b):
    name = get_layer_name()
    return BMMFunction.apply(a, b, cfg, name, _make_emit_fn(state, name, "bmm"))


@_patched("linear", _orig_F_linear)
def _patched_F_linear(state, cfg, input, weight, bias=None):
    name = get_layer_name()
    return LinearFunction.apply(input, weight, bias, cfg, name, _make_emit_fn(state, name, "linear"))


# ---------------------------------------------------------------------------
# SIMD arithmetic (binary)
# ---------------------------------------------------------------------------

@_patched("add", _orig_torch_add)
def _patched_add(state, cfg, a, b, *, alpha=1):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return _orig_torch_add(a, b, alpha=alpha)
    if alpha != 1:
        b = b * alpha
    return SIMDAdd.apply(a, b, _simd_inner_scheme(cfg), True)


@_patched("sub", _orig_torch_sub)
def _patched_sub(state, cfg, a, b, *, alpha=1):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return _orig_torch_sub(a, b, alpha=alpha)
    if alpha != 1:
        b = b * alpha
    return SIMDSub.apply(a, b, _simd_inner_scheme(cfg), True)


@_patched("mul", _orig_torch_mul)
def _patched_mul(state, cfg, a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return _orig_torch_mul(a, b)
    return SIMDMul.apply(a, b, _simd_inner_scheme(cfg), True)


@_patched("div", _orig_torch_div)
def _patched_div(state, cfg, a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return _orig_torch_div(a, b)
    return SIMDDiv.apply(a, b, _simd_inner_scheme(cfg), True)


# ---------------------------------------------------------------------------
# SIMD unary
# ---------------------------------------------------------------------------

@_patched("exp", _orig_torch_exp)
def _patched_exp(state, cfg, x):
    return SIMDExp.apply(x, _simd_inner_scheme(cfg), True)


@_patched("log", _orig_torch_log)
def _patched_log(state, cfg, x):
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
    """Patch torch/F ops and Tensor dunder methods. Returns {key: original}."""
    saved = {}
    for (ns_key, attr), fn in _PATCH_TABLE.items():
        ns = _ns(ns_key)
        saved[(ns_key, attr)] = getattr(ns, attr)
        setattr(ns, attr, fn)

    # Patch Tensor dunder methods (operator overloading bypasses torch.add etc.)
    _patch_tensor_methods(saved)
    return saved


def remove_patches(saved: dict) -> None:
    """Restore torch/F ops and Tensor dunder methods from saved dict."""
    _restore_tensor_methods(saved)
    for key, orig in list(saved.items()):
        if isinstance(key, tuple):
            ns_key, attr = key
            setattr(_ns(ns_key), attr, orig)
        # String keys (Tensor methods) handled by _restore_tensor_methods


# ---------------------------------------------------------------------------
# Tensor dunder method patching
# ---------------------------------------------------------------------------

def _patch_tensor_methods(saved: dict) -> None:
    saved["Tensor.__add__"] = torch.Tensor.__add__
    saved["Tensor.__radd__"] = torch.Tensor.__radd__
    saved["Tensor.__sub__"] = torch.Tensor.__sub__
    saved["Tensor.__rsub__"] = torch.Tensor.__rsub__
    saved["Tensor.__mul__"] = torch.Tensor.__mul__
    saved["Tensor.__rmul__"] = torch.Tensor.__rmul__
    saved["Tensor.__truediv__"] = torch.Tensor.__truediv__
    saved["Tensor.__rtruediv__"] = torch.Tensor.__rtruediv__

    _o_add = _orig_tensor_add
    _o_radd = _orig_tensor_radd
    _o_sub = _orig_tensor_sub
    _o_rsub = _orig_tensor_rsub
    _o_mul = _orig_tensor_mul
    _o_rmul = _orig_tensor_rmul
    _o_div = _orig_tensor_truediv
    _o_rdiv = _orig_tensor_rtruediv

    torch.Tensor.__add__ = lambda s, o: _o_add(s, o) if _is_in_quantize() else _patched_add(s, o)
    torch.Tensor.__radd__ = lambda s, o: _o_radd(s, o) if _is_in_quantize() else _patched_add(o, s)
    torch.Tensor.__sub__ = lambda s, o: _o_sub(s, o) if _is_in_quantize() else _patched_sub(s, o)
    torch.Tensor.__rsub__ = lambda s, o: _o_rsub(s, o) if _is_in_quantize() else _patched_sub(o, s)
    torch.Tensor.__mul__ = lambda s, o: _o_mul(s, o) if _is_in_quantize() else _patched_mul(s, o)
    torch.Tensor.__rmul__ = lambda s, o: _o_rmul(s, o) if _is_in_quantize() else _patched_mul(o, s)
    torch.Tensor.__truediv__ = lambda s, o: _o_div(s, o) if _is_in_quantize() else _patched_div(s, o)
    torch.Tensor.__rtruediv__ = lambda s, o: _o_rdiv(s, o) if _is_in_quantize() else _patched_div(o, s)


def _restore_tensor_methods(saved: dict) -> None:
    for attr in ("__add__", "__radd__", "__sub__", "__rsub__",
                 "__mul__", "__rmul__", "__truediv__", "__rtruediv__"):
        key = f"Tensor.{attr}"
        if key in saved:
            setattr(torch.Tensor, attr, saved[key])
