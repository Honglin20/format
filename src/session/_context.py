"""QuantizeContext: context manager for inline-op quantization interception.

Merged from context/_state.py + context/_stack.py + context/quantize_context.py.
"""
import contextvars
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.scheme.op_config import OpQuantConfig


# ══════════════════════════════════════════════════════════════════════════════
# State (was: _state.py)
# ══════════════════════════════════════════════════════════════════════════════

_EMPTY_CFG = OpQuantConfig()


@dataclass
class _CtxState:
    cfg: OpQuantConfig
    op_cfgs: Dict[str, OpQuantConfig] = field(default_factory=dict)
    observers: List = field(default_factory=list)

    def resolve(self, op_name: str) -> OpQuantConfig:
        if self.op_cfgs and op_name in self.op_cfgs:
            return self.op_cfgs[op_name]
        return self.cfg


_ctx_state: contextvars.ContextVar[Optional[_CtxState]] = contextvars.ContextVar(
    "quant_ctx_state", default=None
)


# Observer-only mode: lightweight state for inline-op observer tracking during
# analysis.  When set, patched torch/F ops emit observer events WITHOUT
# re-quantizing — they use the original function and record the result.
@dataclass
class _ObserverState:
    observers: List = field(default_factory=list)


_observer_state: contextvars.ContextVar[Optional[_ObserverState]] = (
    contextvars.ContextVar("observer_state", default=None)
)


# ══════════════════════════════════════════════════════════════════════════════
# Module stack (was: _stack.py)
# ══════════════════════════════════════════════════════════════════════════════

_module_stack: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "quant_module_stack", default=None
)


# ──────────────────────────────────────────────────────────────────────────────
# Export scales (for ONNX QDQ node scale wiring)
# ──────────────────────────────────────────────────────────────────────────────

_export_scales_var: contextvars.ContextVar[Optional[Dict[str, torch.Tensor]]] = (
    contextvars.ContextVar("quant_export_scales", default=None)
)


def _collect_export_scales(model) -> Dict[str, "torch.Tensor"]:
    """Walk model and collect ``_output_scale`` buffers keyed by module name.

    Returns an empty dict if no scales are found (model not calibrated).
    """
    scales: Dict[str, "torch.Tensor"] = {}
    for name, module in model.named_modules():
        if hasattr(module, "_output_scale"):
            scales[name] = module._output_scale
    return scales


# ContextVar for the current module's output_scale during ONNX symbolic tracing.
# Set by LinearFunction.symbolic() (and similar) before emitting QDQ nodes,
# read by FormatBase.export_onnx() subclasses to embed calibrated scale values
# instead of placeholder 1.0.
_onnx_current_scale_var: contextvars.ContextVar[Optional[torch.Tensor]] = (
    contextvars.ContextVar("quant_onnx_current_scale", default=None)
)


def get_layer_name() -> str:
    stack = _module_stack.get()
    return ".".join(stack) if stack else ""


def _make_pre_hook(name: str):
    def _pre(module, inp):
        stack = list(_module_stack.get() or [])
        stack.append(name)
        _module_stack.set(stack)
    return _pre


def _make_post_hook(name: str):
    def _post(module, inp, out):
        stack = list(_module_stack.get() or [])
        if stack and stack[-1] == name:
            stack.pop()
            _module_stack.set(stack)
    return _post


def install_stack_hooks(model: nn.Module) -> List:
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_pre_hook(_make_pre_hook(name)))
        handles.append(
            module.register_forward_hook(_make_post_hook(name), always_call=True)
        )
    return handles


def remove_stack_hooks(handles: List) -> None:
    for h in handles:
        h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# QuantizeContext public API (was: quantize_context.py)
# ══════════════════════════════════════════════════════════════════════════════

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
        observers: Optional observers.
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
        from src.session._patches import apply_patches

        self._ctx_token = _ctx_state.set(self._state)
        try:
            self._hook_handles = install_stack_hooks(self.model)
            self._saved_ops = apply_patches()
        except:
            _ctx_state.reset(self._ctx_token)
            self._ctx_token = None
            raise
        return self

    def __exit__(self, *args):
        from src.session._patches import remove_patches

        try:
            remove_patches(self._saved_ops)
        finally:
            try:
                remove_stack_hooks(self._hook_handles)
            finally:
                if self._ctx_token is not None:
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
        if self._ctx_token is None:
            raise RuntimeError(
                "QuantizeContext.export_onnx() must be called inside a 'with' block"
            )
        from src.onnx.export import export_quantized_model

        export_scales = _collect_export_scales(self.model)
        _export_scales_var.set(export_scales if export_scales else None)
        try:
            export_quantized_model(self.model, dummy_input, output_path, opset_version)
        finally:
            _export_scales_var.set(None)
