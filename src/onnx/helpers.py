"""
ONNX export helper utilities.

_is_standard_format: int8/int4/int2/fp8 + non-PER_BLOCK → True (standard QDQ).
_emit_quantize_node: emit QuantizeLinear/DequantizeLinear or MxQuantize node.
"""
import torch
from src.scheme.granularity import GranularityMode

# Formats that map to ONNX standard QDQ nodes (opset 13+).
# Per-block variants of these formats are excluded (MX block style → custom op).
_STANDARD_NAMES = {"int8", "int4", "int2", "fp8_e4m3", "fp8_e5m2"}


def _is_standard_format(scheme) -> bool:
    """Return True if scheme should export as ONNX QDQ (QuantizeLinear/DequantizeLinear).

    Rules:
    - PER_BLOCK granularity → always False (MX block quantization → custom op)
    - int8/int4/int2/fp8_e4m3/fp8_e5m2 + per_tensor or per_channel → True
    - All other formats → False (custom op)
    """
    if scheme.granularity.mode == GranularityMode.PER_BLOCK:
        return False
    return scheme.format.name in _STANDARD_NAMES


def _emit_quantize_node(g, x, scheme):
    """Emit a quantize+dequantize pair in the ONNX graph for the given scheme.

    Delegates to scheme.format.export_onnx() — each format controls its own
    ONNX representation (Strategy pattern, consistent with FormatBase.quantize()).

    Standard formats → QuantizeLinear(x, scale=1.0, zp=0) → DequantizeLinear.
    Non-standard / MX formats → com.microxscaling::MxQuantize custom node.
    """
    return scheme.format.export_onnx(g, x, scheme)
