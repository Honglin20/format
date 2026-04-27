"""
quantized_bmm: OpQuantConfig-driven replacement for mx.bmm.

Forward + backward are bit-exact equivalent to mx/bmm.py when driven by
the same OpQuantConfig produced by op_config_from_mx_specs.

BMMFunction always uses a_elem_format for both inputs (no mode_config).
No bias support (matches mx/bmm.py).
"""
import torch

from src.scheme.op_config import OpQuantConfig

_torch_bmm = torch.bmm
from src.quantize import quantize


class BMMFunction(torch.autograd.Function):
    """Autograd function for quantized bmm with QAT backward.

    Forward flow (matches mx/bmm.py):
      1. storage quantize in1, in2
      2. compute quantize in1 along last dim, in2 along second-to-last
      3. torch.bmm(qin1, qin2)
      4. storage + compute quantize output

    No bias (matches mx/bmm.py).
    """

    @staticmethod
    def forward(ctx, in1, in2, cfg: OpQuantConfig, name=None, emit_fn=None):
        ctx.emit_fn = emit_fn
        in1_raw, in2_raw = in1, in2

        # in1: storage → compute
        if cfg.storage is not None:
            fp_in1 = in1; in1 = quantize(in1, cfg.storage)
            if emit_fn: emit_fn("input", 0, "input_pre_quant", fp_in1, in1, cfg.storage)
        in1_post_storage = in1
        if cfg.input is not None:
            fp_in1 = in1; in1 = quantize(in1, cfg.input)
            if emit_fn: emit_fn("input", 1, "input_pre_quant", fp_in1, in1, cfg.input)

        # in2: storage → compute
        if cfg.storage is not None:
            fp_in2 = in2; in2 = quantize(in2, cfg.storage)
            if emit_fn: emit_fn("weight", 0, "weight_pre_quant", fp_in2, in2, cfg.storage)
        in2_post_storage = in2
        if cfg.weight is not None:
            fp_in2 = in2; in2 = quantize(in2, cfg.weight)
            if emit_fn: emit_fn("weight", 1, "weight_pre_quant", fp_in2, in2, cfg.weight)

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(in1_post_storage, in2_post_storage)
        else:
            ctx.save_for_backward(in1_raw, in2_raw)

        ctx.cfg = cfg
        ctx.name = name

        # Compute bmm
        out = _torch_bmm(in1, in2)

        # Output: storage
        if cfg.storage is not None:
            fp_out = out; out = quantize(out, cfg.storage)
            if emit_fn: emit_fn("output", 0, "output_post_quant", fp_out, out, cfg.storage)

        # Output compute
        if cfg.output is not None:
            fp_out = out; out = quantize(out, cfg.output)
            if emit_fn: emit_fn("output", 1, "output_post_quant", fp_out, out, cfg.output)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        in1, in2 = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        emit_fn = ctx.emit_fn

        # grad_output: storage → compute
        if cfg.storage is not None:
            fp_go = grad_out; grad_out = quantize(grad_out, cfg.storage)
            if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_go, grad_out, cfg.storage)
        if cfg.grad_output is not None:
            fp_go = grad_out; grad_out = quantize(grad_out, cfg.grad_output)
            if emit_fn: emit_fn("grad_output", 1, "grad_output_pre_quant", fp_go, grad_out, cfg.grad_output)

        # grad_in2: in1^T @ grad_out
        in1_gw = in1
        if cfg.storage is not None:
            in1_gw = quantize(in1_gw, cfg.storage)
        if cfg.input_gw is not None:
            in1_gw = quantize(in1_gw, cfg.input_gw)

        g_gw = grad_out
        if cfg.storage is not None:
            g_gw = quantize(g_gw, cfg.storage)
        if cfg.grad_output_gw is not None:
            g_gw = quantize(g_gw, cfg.grad_output_gw)

        grad_in2 = _torch_bmm(in1_gw.transpose(-1, -2), g_gw)

        if cfg.storage is not None:
            fp_gw = grad_in2; grad_in2 = quantize(grad_in2, cfg.storage)
            if emit_fn: emit_fn("grad_weight", 0, "grad_weight_post_quant", fp_gw, grad_in2, cfg.storage)
        if cfg.grad_weight is not None:
            fp_gw = grad_in2; grad_in2 = quantize(grad_in2, cfg.grad_weight)
            if emit_fn: emit_fn("grad_weight", 1, "grad_weight_post_quant", fp_gw, grad_in2, cfg.grad_weight)

        # grad_in1: grad_out @ in2^T
        in2_gi = in2
        if cfg.storage is not None:
            in2_gi = quantize(in2_gi, cfg.storage)
        if cfg.weight_gi is not None:
            in2_gi = quantize(in2_gi, cfg.weight_gi)

        g_gi = grad_out
        if cfg.storage is not None:
            g_gi = quantize(g_gi, cfg.storage)
        if cfg.grad_output_gi is not None:
            g_gi = quantize(g_gi, cfg.grad_output_gi)

        grad_in1 = _torch_bmm(g_gi, in2_gi.transpose(-1, -2))

        if cfg.storage is not None:
            fp_gi = grad_in1; grad_in1 = quantize(grad_in1, cfg.storage)
            if emit_fn: emit_fn("grad_input", 0, "grad_input_post_quant", fp_gi, grad_in1, cfg.storage)
        if cfg.grad_input is not None:
            fp_gi = grad_in1; grad_in1 = quantize(grad_in1, cfg.grad_input)
            if emit_fn: emit_fn("grad_input", 1, "grad_input_post_quant", fp_gi, grad_in1, cfg.grad_input)

        return grad_in1, grad_in2, None, None, None

    @staticmethod
    def symbolic(g, in1, in2, cfg, name, emit_fn=None):
        from src.onnx.helpers import _emit_quantize_node

        if cfg.storage is not None:
            in1 = _emit_quantize_node(g, in1, cfg.storage)
        if cfg.input is not None:
            in1 = _emit_quantize_node(g, in1, cfg.input)

        if cfg.storage is not None:
            in2 = _emit_quantize_node(g, in2, cfg.storage)
        if cfg.weight is not None:
            in2 = _emit_quantize_node(g, in2, cfg.weight)

        out = g.op("MatMul", in1, in2)

        if cfg.storage is not None:
            out = _emit_quantize_node(g, out, cfg.storage)
        if cfg.output is not None:
            out = _emit_quantize_node(g, out, cfg.output)

        return out


def quantized_bmm(in1, in2, cfg=None, name=None):
    """Functional API: quantized bmm with OpQuantConfig."""
    if cfg is None or cfg == OpQuantConfig():
        return _torch_bmm(in1, in2)

    return BMMFunction.apply(in1, in2, cfg, name)
