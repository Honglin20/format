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
        try:
            self._hook_handles = install_stack_hooks(self.model)
            self._saved_ops = apply_patches()
        except:
            _ctx_state.reset(self._ctx_token)
            self._ctx_token = None
            raise
        return self

    def __exit__(self, *args):
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
