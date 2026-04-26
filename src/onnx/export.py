"""
export_quantized_model: export a quantized model to ONNX.

Wraps torch.onnx.export with com.microxscaling custom opset registration
and verifies the output graph with onnx.checker.
"""
import torch
import torch.nn as nn


def export_quantized_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export a quantized model to an ONNX file.

    Args:
        model: Module containing QuantizedLinear / QuantizedConv{1,2,3}d layers.
            Must have symbolic() methods on its autograd.Function subclasses
            (added in Phase 5).
        dummy_input: Representative input tensor (defines input shape in graph).
        output_path: Path to write the .onnx file.
        opset_version: ONNX opset version. Default: 17.

    The exported graph uses:
    - QuantizeLinear/DequantizeLinear for int8/int4/int2/fp8 formats.
    - com.microxscaling::MxQuantize for MX block-format quantization.

    Note: Scale values in QDQ nodes are placeholder constants (1.0);
    the graph is valid for visualization but not for runtime inference.
    """
    torch.onnx.export(
        model,
        (dummy_input,),
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
