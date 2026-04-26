"""
quantized_matmul: OpQuantConfig-driven replacement for mx.matmul.

Forward + backward are bit-exact equivalent to mx/matmul.py when driven by
the same OpQuantConfig produced by op_config_from_mx_specs.

MatMulFunction supports mode_config ('aa', 'aw', 'wa') which determines
which elem_format is used for each input in the MX block quantization step.
"""
import numpy as np
import torch

from src.scheme.op_config import OpQuantConfig

_torch_matmul = torch.matmul
from src.quantize import quantize
from src.scheme.granularity import GranularityMode


class MatMulFunction(torch.autograd.Function):
    """Autograd function for quantized matmul with QAT backward.

    Forward flow (matches mx/matmul.py):
      1. elemwise quantize in1, in2
      2. MX quantize in1 along last dim, in2 along second-to-last dim
      3. torch.matmul(qin1, qin2)
      4. elemwise quantize output (post-matmul)
      5. if bias: add quantized bias + elemwise quantize output (post-bias)

    The mode_config ('aa', 'aw', 'wa') determines which input gets
    a_elem_format vs w_elem_format for MX quantization.
    """

    @staticmethod
    def forward(ctx, in1, in2, bias, cfg: OpQuantConfig, name=None, mode_config='aa', emit_fn=None):
        assert mode_config in ("aa", "aw", "wa")
        ctx.mode_config = mode_config
        ctx.emit_fn = emit_fn

        # Save raw tensors for STE backward
        in1_raw, in2_raw = in1, in2

        # --- Split input pipeline into elemwise + MX ---
        in1_elem = tuple(s for s in cfg.input if s.granularity.mode != GranularityMode.PER_BLOCK)
        in1_mx = tuple(s for s in cfg.input if s.granularity.mode == GranularityMode.PER_BLOCK)

        # --- Split weight pipeline into elemwise + MX ---
        in2_elem = tuple(s for s in cfg.weight if s.granularity.mode != GranularityMode.PER_BLOCK)
        in2_mx = tuple(s for s in cfg.weight if s.granularity.mode == GranularityMode.PER_BLOCK)

        # Elemwise quantize in1 and in2
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

        # MX quantize in1 (along last dim, axis=-1) — already covered in the MX loop above
        # MX quantize in2 (along second-to-last dim, axis=-2)
        for s in in2_mx:
            fp_in2 = in2
            in2 = quantize(in2, s)
            if emit_fn: emit_fn("weight", wt_idx, "weight_pre_quant", fp_in2, in2, s)
            wt_idx += 1

        # Bias quantization
        q_bias = None
        if bias is not None:
            ctx.bias_shape = list(bias.shape)
            q_bias = bias
            b_idx = 0
            for s in cfg.bias:
                fp_b = q_bias
                q_bias = quantize(q_bias, s)
                if emit_fn: emit_fn("bias", b_idx, "weight_pre_quant", fp_b, q_bias, s)
                b_idx += 1
        else:
            ctx.bias_shape = None

        # Save for backward
        if cfg.is_training:
            ctx.save_for_backward(in1_post_elem, in2_post_elem)
        else:
            ctx.save_for_backward(in1_raw, in2_raw)

        ctx.cfg = cfg
        ctx.name = name

        # Compute matmul
        out = _torch_matmul(in1, in2)

        # Output quantization step 1 (post-matmul)
        out_idx = 0
        if len(cfg.output) > 0:
            fp_out = out
            out = quantize(out, cfg.output[0])
            if emit_fn: emit_fn("output", out_idx, "output_post_quant", fp_out, out, cfg.output[0])
            out_idx += 1

        # Add bias + output quantization step 2 (post-bias)
        if q_bias is not None:
            out = out + q_bias
            if len(cfg.output) > 1:
                fp_out = out
                out = quantize(out, cfg.output[1])
                if emit_fn: emit_fn("output", out_idx, "output_post_quant", fp_out, out, cfg.output[1])
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

        # --- grad_in1: grad_out @ in2^T ---
        # grad_in2: in1^T @ grad_out ---

        # input_gw (in1 for grad_in2 gemm): MX along axis=-2
        in1_for_grad_in2 = in1
        for s in cfg.input_gw:
            in1_for_grad_in2 = quantize(in1_for_grad_in2, s)

        # grad_output_gw (grad_out for grad_in2 gemm): MX along axis=-2
        g_for_grad_in2 = grad_out
        for s in cfg.grad_output_gw:
            g_for_grad_in2 = quantize(g_for_grad_in2, s)

        # weight_gi (in2 for grad_in1 gemm): MX along axis=-1
        in2_for_grad_in1 = in2
        for s in cfg.weight_gi:
            in2_for_grad_in1 = quantize(in2_for_grad_in1, s)

        # grad_output_gi (grad_out for grad_in1 gemm): MX along axis=-1
        g_for_grad_in1 = grad_out
        for s in cfg.grad_output_gi:
            g_for_grad_in1 = quantize(g_for_grad_in1, s)

        # grad_in1 = grad_out @ in2^T
        grad_in1 = _torch_matmul(g_for_grad_in1, in2_for_grad_in1.transpose(-1, -2))

        # grad_in2 = in1^T @ grad_out
        grad_in2 = _torch_matmul(in1_for_grad_in2.transpose(-1, -2), g_for_grad_in2)

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

        # grad_bias
        grad_bias = None
        if ctx.bias_shape is not None:
            inner_size = grad_out.shape[-1]
            assert np.prod(ctx.bias_shape) == inner_size
            grad_bias = grad_out.reshape(-1, inner_size).sum(0)
            grad_bias = grad_bias.reshape(ctx.bias_shape)
            for s in cfg.grad_bias:
                grad_bias = quantize(grad_bias, s)

        return grad_in1, grad_in2, grad_bias, None, None, None, None

    @staticmethod
    def symbolic(g, in1, in2, bias, cfg, name, mode_config, emit_fn):
        """ONNX symbolic: Q/DQ wrappers + MatMul + optional Add."""
        from src.onnx.helpers import _emit_quantize_node
        for scheme in cfg.input:
            in1 = _emit_quantize_node(g, in1, scheme)
        for scheme in cfg.weight:
            in2 = _emit_quantize_node(g, in2, scheme)
        out = g.op("MatMul", in1, in2)
        if len(cfg.output) > 0:
            out = _emit_quantize_node(g, out, cfg.output[0])
        if bias is not None:
            qb = bias
            for scheme in cfg.bias:
                qb = _emit_quantize_node(g, qb, scheme)
            out = g.op("Add", out, qb)
            if len(cfg.output) > 1:
                out = _emit_quantize_node(g, out, cfg.output[1])
        return out


def quantized_matmul(in1, in2, bias=None, cfg=None, name=None, mode_config='aa'):
    """Functional API: quantized matmul with OpQuantConfig."""
    if cfg is None or cfg == OpQuantConfig():
        if bias is None:
            return _torch_matmul(in1, in2)
        return torch.addmm(bias, in1, in2)

    return MatMulFunction.apply(in1, in2, bias, cfg, name, mode_config)
