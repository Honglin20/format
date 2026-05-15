"""
export_quantized_model: export a quantized model to ONNX.

Wraps torch.onnx.export with com.microxscaling custom opset registration
and verifies the output graph with onnx.checker.
"""
import torch
import torch.nn as nn

from src.session._context import _onnx_export_active


def export_quantized_model(
    model: nn.Module,
    dummy_input,  # Tensor | tuple | list | dict
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export a quantized model to an ONNX file.

    Args:
        model: Module containing QuantizedLinear / QuantizedConv{1,2,3}d layers.
            Must have symbolic() methods on its autograd.Function subclasses
            (added in Phase 5).
        dummy_input: Representative input (Tensor, tuple, list, or dict).
        output_path: Path to write the .onnx file.
        opset_version: ONNX opset version. Default: 17.

    The exported graph uses unified three-axis decomposition:
    - com.microxscaling::Scale for granularity (per_tensor, per_channel, per_block).
    - com.microxscaling::Quantize for element-wise format quantization.
    - com.microxscaling::Truncate for bf16/fp16 truncation (no Scale needed).

    Scale nodes carry calibration-derived values when available;
    otherwise they use placeholder values (valid for visualization only).
    """
    # Handle multi-input types: torch.onnx.export expects:
    # - tensor: wrapped in tuple → model(tensor)
    # - tuple / list: converted to tuple → model(*args)
    # - dict: kept as-is → model(**kwargs)
    if isinstance(dummy_input, (tuple, list)):
        args = tuple(dummy_input)
    elif isinstance(dummy_input, dict):
        args = dummy_input
    else:
        args = (dummy_input,)

    # Swap model.forward back to the original (pre-quantize_model) forward
    # so the ONNX tracer sees the real module graph without QuantizeContext
    # wrapping.  QuantizeContext construction involves ContextVar, _CtxState
    # dataclass, and torch/F patch table mutations — none of which Dynamo
    # can trace.  We save/restore the wrapped forward around the export call.
    _orig_fwd = model.forward
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward

    # Signal that we are inside ONNX export so per-block quantization can be
    # skipped during tracing (both TorchScript and Dynamo paths).  The
    # Function.symbolic() methods emit the equivalent ONNX nodes.
    _token = _onnx_export_active.set(True)
    try:
        # Prefer the TorchScript-based exporter (dynamo=False) when available
        # (PyTorch >= 2.5).  The Dynamo-based exporter cannot trace through
        # our custom autograd.Function subclasses because they take non-Tensor
        # arguments (OpQuantConfig, strings, etc.) — Dynamo's HigherOrderOperator
        # wrapper requires all inputs to be Tensors.  The TorchScript path
        # calls Function.symbolic() directly, which emits clean QDQ/custom
        # ONNX nodes and works correctly.
        import inspect
        _export_kwargs = dict(
            opset_version=opset_version,
            custom_opsets={"com.microxscaling": 1},
            do_constant_folding=False,
        )
        if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
            _export_kwargs['dynamo'] = False
        torch.onnx.export(model, args, output_path, **_export_kwargs)
    finally:
        _onnx_export_active.reset(_token)
        model.forward = _orig_fwd
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
