"""
quantized_bmm: OpQuantConfig-driven replacement for mx.bmm.

Forward + backward are bit-exact equivalent to mx/bmm.py when driven by
the same OpQuantConfig produced by op_config_from_mx_specs.

BMMFunction always uses a_elem_format for both inputs (no mode_config).
No bias support (matches mx/bmm.py).
"""
import torch

from src.scheme.op_config import OpQuantConfig
from src.quantize import quantize
from src.scheme.granularity import GranularityMode


class BMMFunction(torch.autograd.Function):
    """Autograd function for quantized bmm with QAT backward.

    Forward flow (matches mx/bmm.py):
      1. elemwise quantize in1, in2
      2. MX quantize in1 along last dim (axis=-1), in2 along second-to-last (axis=-2)
      3. torch.bmm(qin1, qin2)
      4. elemwise quantize output

    No bias (matches mx/bmm.py).
    """

    @staticmethod
    def forward(ctx, in1, in2, cfg: OpQuantConfig, name=None, emit_fn=None):
        ctx.emit_fn = emit_fn
        in1_raw, in2_raw = in1, in2

        # --- Split input pipeline into elemwise + MX ---
        in1_elem = tuple(s for s in cfg.input if s.granularity.mode != GranularityMode.PER_BLOCK)
        in1_mx = tuple(s for s in cfg.input if s.granularity.mode == GranularityMode.PER_BLOCK)

        in2_elem = tuple(s for s in cfg.weight if s.granularity.mode != GranularityMode.PER_BLOCK)
        in2_mx = tuple(s for s in cfg.weight if s.granularity.mode == GranularityMode.PER_BLOCK)

        # Elemwise quantize
        in_idx = 0
        for s in in1_elem:
            fp_in1 = in1
            in1 = quantize(in1, s)
            if emit_fn: emit_fn("input", in_idx, "input_pre_quant", fp_in1, in1, s)
            in_idx += 1
        in1_post_elem = in1

        for s in in1_mx:
            fp_in1 = in1
            in1 = quantize(in1, s)
            if emit_fn: emit_fn("input", in_idx, "input_pre_quant", fp_in1, in1, s)
            in_idx += 1

        wt_idx = 0
        for s in in2_elem:
            fp_in2 = in2
            in2 = quantize(in2, s)
            if emit_fn: emit_fn("weight", wt_idx, "weight_pre_quant", fp_in2, in2, s)
            wt_idx += 1
        in2_post_elem = in2

        # MX quantize
        for s in in2_mx:
            fp_in2 = in2
            in2 = quantize(in2, s)
            if emit_fn: emit_fn("weight", wt_idx, "weight_pre_quant", fp_in2, in2, s)
            wt_idx += 1

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(in1_post_elem, in2_post_elem)
        else:
            ctx.save_for_backward(in1_raw, in2_raw)

        ctx.cfg = cfg
        ctx.name = name

        # Compute bmm
        out = torch.bmm(in1, in2)

        # Output quantization (single elemwise step, no bias)
        out_idx = 0
        if len(cfg.output) > 0:
            fp_out = out
            out = quantize(out, cfg.output[0])
            if emit_fn: emit_fn("output", out_idx, "output_post_quant", fp_out, out, cfg.output[0])
            out_idx += 1

        return out

    @staticmethod
    def backward(ctx, grad_out):
        in1, in2 = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        emit_fn = ctx.emit_fn

        # Quantize grad_output
        go_idx = 0
        for s in cfg.grad_output:
            fp_go = grad_out
            grad_out = quantize(grad_out, s)
            if emit_fn: emit_fn("grad_output", go_idx, "grad_output_pre_quant", fp_go, grad_out, s)
            go_idx += 1

        # in1 for grad_in2: MX along axis=-2
        in1_for_grad_in2 = in1
        for s in cfg.input_gw:
            in1_for_grad_in2 = quantize(in1_for_grad_in2, s)

        # grad_out for grad_in2: MX along axis=-2
        g_for_grad_in2 = grad_out
        for s in cfg.grad_output_gw:
            g_for_grad_in2 = quantize(g_for_grad_in2, s)

        # in2 for grad_in1: MX along axis=-1
        in2_for_grad_in1 = in2
        for s in cfg.weight_gi:
            in2_for_grad_in1 = quantize(in2_for_grad_in1, s)

        # grad_out for grad_in1: MX along axis=-1
        g_for_grad_in1 = grad_out
        for s in cfg.grad_output_gi:
            g_for_grad_in1 = quantize(g_for_grad_in1, s)

        # grad_in1 = grad_out @ in2^T
        grad_in1 = torch.bmm(g_for_grad_in1, in2_for_grad_in1.transpose(-1, -2))

        # grad_in2 = in1^T @ grad_out
        grad_in2 = torch.bmm(in1_for_grad_in2.transpose(-1, -2), g_for_grad_in2)

        # Exit elemwise quantize
        gi_idx = 0
        for s in cfg.grad_input:
            fp_gi = grad_in1
            grad_in1 = quantize(grad_in1, s)
            if emit_fn: emit_fn("grad_input", gi_idx, "grad_input_post_quant", fp_gi, grad_in1, s)
            gi_idx += 1

        gw_idx = 0
        for s in cfg.grad_weight:
            fp_gw = grad_in2
            grad_in2 = quantize(grad_in2, s)
            if emit_fn: emit_fn("grad_weight", gw_idx, "grad_weight_post_quant", fp_gw, grad_in2, s)
            gw_idx += 1

        return grad_in1, grad_in2, None, None, None


def quantized_bmm(in1, in2, cfg=None, name=None):
    """Functional API: quantized bmm with OpQuantConfig."""
    if cfg is None or cfg == OpQuantConfig():
        return torch.bmm(in1, in2)

    return BMMFunction.apply(in1, in2, cfg, name)
