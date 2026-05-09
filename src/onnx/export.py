"""
export_quantized_model: export a quantized model to ONNX.

Wraps torch.onnx.export with com.microxscaling custom opset registration
and verifies the output graph with onnx.checker.
"""
import torch
import torch.nn as nn


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

    The exported graph uses:
    - QuantizeLinear/DequantizeLinear for int8/int4/fp8 per_tensor/per_channel formats.
    - com.microxscaling::MxQuantize for MX per_block and truncation formats.
    - com.microxscaling::NF4Quantize for NF4 lookup-table format.
    - int2 uses MxQuantize (ONNX QDQ does not support 2-bit integer types).

    Scale values in QDQ nodes reflect calibration results when available;
    otherwise they default to 1.0 (valid for visualization only).
    """
    # Handle multi-input types: torch.onnx.export expects a tuple of positional
    # arguments; single Tensor gets wrapped, tuple/list are used as-is (list
    # converted for safety), dict is passed as a single arg.
    if isinstance(dummy_input, (tuple, list)):
        args = tuple(dummy_input)
    else:
        args = (dummy_input,)
    torch.onnx.export(
        model,
        args,
        output_path,
        opset_version=opset_version,
        custom_opsets={"com.microxscaling": 1},
        do_constant_folding=False,
    )
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
