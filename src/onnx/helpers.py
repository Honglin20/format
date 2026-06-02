"""
ONNX export helpers — unified three-axis (format × granularity × transform).

Every quantization emits up to three nodes:
  1. Scale node         — granularity axis (how scale is shared)
  2. Quantize node      — format axis (element-wise quantization levels)
  3. Transform node     — transform axis (optional, TBD)

The graph pattern is always:
  x → Scale(mode, ...) → Quantize(format, bits/ebits/mbits/levels) → out

For truncation formats (bf16, fp16), Scale is not needed — Truncate is used directly.
"""
import torch
from src.scheme.granularity import GranularityMode


def _emit_scale_node(g, x, granularity):
    """Emit a Scale node representing the granularity axis.

    Returns the scale-tensor output that feeds into the Quantize node.
    """
    mode = granularity.mode
    if mode == GranularityMode.PER_TENSOR:
        return g.op("com.microxscaling::Scale", x,
                    mode_s="per_tensor")
    elif mode == GranularityMode.PER_CHANNEL:
        return g.op("com.microxscaling::Scale", x,
                    mode_s="per_channel",
                    axis_i=granularity.channel_axis)
    elif mode == GranularityMode.PER_BLOCK:
        return g.op("com.microxscaling::Scale", x,
                    mode_s="per_block",
                    block_size_i=granularity.block_size,
                    axis_i=granularity.block_axis)
    raise ValueError(f"Unknown granularity mode: {mode}")


def _emit_format_node(g, x, scale, format_obj):
    """Emit a Quantize node representing the format axis.

    The Quantize node does: normalize by scale → quantize to format levels
    → rescale by scale (i.e. a full quantize+dequantize round-trip).
    """
    from src.formats.lookup_formats import LookupFormat
    from src.formats.bf16_fp16 import BFloat16Format, Float16Format

    if isinstance(format_obj, BFloat16Format):
        return _emit_truncate(g, x, "bfloat16")
    if isinstance(format_obj, Float16Format):
        return _emit_truncate(g, x, "float16")
    if isinstance(format_obj, LookupFormat):
        levels = format_obj.levels.detach().cpu().tolist()
        return g.op("com.microxscaling::Quantize", x, scale,
                    format_s=format_obj.name,
                    ebits_i=format_obj.ebits,
                    mbits_i=format_obj.mbits,
                    levels_f=levels)
    return g.op("com.microxscaling::Quantize", x, scale,
                format_s=format_obj.name,
                ebits_i=format_obj.ebits,
                mbits_i=format_obj.mbits)


def _emit_truncate(g, x, dtype):
    """Emit a Truncate node for bf16/fp16 — no scale needed."""
    return g.op("com.microxscaling::Truncate", x, dtype_s=dtype)


def _emit_quantize_node(g, x, scheme):
    """Emit the unified three-axis node sequence for a QuantScheme.

    Delegates to scheme.format.export_onnx() which produces:
      Scale(granularity) → Quantize/Truncate(format) → output
    """
    return scheme.format.export_onnx(g, x, scheme)


def _emit_binary_onnx(g, in1, in2, inner_scheme, op_name):
    """Emit a binary op with optional quantize wrappers on inputs and output."""
    if inner_scheme is not None:
        in1 = _emit_quantize_node(g, in1, inner_scheme)
        if isinstance(in2, torch._C.Value):
            in2 = _emit_quantize_node(g, in2, inner_scheme)
    out = g.op(op_name, in1, in2)
    if inner_scheme is not None:
        out = _emit_quantize_node(g, out, inner_scheme)
    return out


def _emit_unary_onnx(g, in1, inner_scheme, op_name):
    """Emit a unary op with optional quantize wrappers on input and output."""
    if inner_scheme is not None:
        in1 = _emit_quantize_node(g, in1, inner_scheme)
    out = g.op(op_name, in1)
    if inner_scheme is not None:
        out = _emit_quantize_node(g, out, inner_scheme)
    return out
