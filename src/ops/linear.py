"""
QuantizedLinear: OpQuantConfig-driven replacement for mx.Linear.

Forward + backward are bit-exact equivalent to mx/linear.py when driven by
the same OpQuantConfig produced by op_config_from_mx_specs.

Key design: forward saves post-elemwise (pre-MX) tensors for backward,
matching mx/linear.py which saves bf_in/bf_weight (elemwise-only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scheme.op_config import OpQuantConfig

_F_linear = F.linear
from src.quantize import quantize
from src.analysis.mixin import ObservableMixin
from src.scheme.granularity import GranularityMode


class LinearFunction(torch.autograd.Function):
    """Autograd function for quantized linear with QAT backward.

    Forward flow (matches mx/linear.py):
      1. elemwise quantize input (save post-elemwise for backward)
      2. MX quantize input
      3. elemwise quantize weight (save post-elemwise for backward)
      4. MX quantize weight
      5. elemwise quantize bias
      6. F.linear(qinput, qweight) — no bias
      7. elemwise quantize output[0] (post-matmul, pre-bias)
      8. add quantized bias
      9. elemwise quantize output[1] (post-bias-add)

    Backward flow:
      Saved tensors are post-elemwise (matching mx/linear.py's bf_in/bf_weight).
      MX schemes in input_gw/grad_output_gw/weight_gi/grad_output_gi apply to
      already-elemwise-quantized tensors — no double elemwise.
    """

    @staticmethod
    def forward(ctx, x, w, b, cfg: OpQuantConfig, name=None, emit_fn=None,
                output_scale=None):
        ctx.emit_fn = emit_fn
        # Save raw tensors for STE backward (when is_training=False)
        x_raw, w_raw = x, w

        # --- Input pipeline: elemwise first, then MX ---
        input_elem = tuple(s for s in cfg.input if s.granularity.mode != GranularityMode.PER_BLOCK)
        input_mx = tuple(s for s in cfg.input if s.granularity.mode == GranularityMode.PER_BLOCK)

        in_idx = 0
        for s in input_elem:
            fp_x = x
            x = quantize(x, s)
            if emit_fn: emit_fn("input", in_idx, "input_pre_quant", fp_x, x, s)
            in_idx += 1
        x_post_elem = x  # save intermediate for backward

        for s in input_mx:
            fp_x = x
            x = quantize(x, s)
            if emit_fn: emit_fn("input", in_idx, "input_pre_quant", fp_x, x, s)
            in_idx += 1

        # --- Weight pipeline: elemwise first, then MX ---
        weight_elem = tuple(s for s in cfg.weight if s.granularity.mode != GranularityMode.PER_BLOCK)
        weight_mx = tuple(s for s in cfg.weight if s.granularity.mode == GranularityMode.PER_BLOCK)

        wt_idx = 0
        for s in weight_elem:
            fp_w = w
            w = quantize(w, s)
            if emit_fn: emit_fn("weight", wt_idx, "weight_pre_quant", fp_w, w, s)
            wt_idx += 1
        w_post_elem = w

        for s in weight_mx:
            fp_w = w
            w = quantize(w, s)
            if emit_fn: emit_fn("weight", wt_idx, "weight_pre_quant", fp_w, w, s)
            wt_idx += 1

        # --- Bias pipeline (elemwise only, no MX) ---
        q_bias = None
        if b is not None:
            q_bias = b
            b_idx = 0
            for s in cfg.bias:
                fp_b = q_bias
                q_bias = quantize(q_bias, s)
                if emit_fn: emit_fn("bias", b_idx, "weight_pre_quant", fp_b, q_bias, s)
                b_idx += 1

        # Save for backward: post-elemwise if training, raw if STE
        if cfg.is_training:
            ctx.save_for_backward(x_post_elem, w_post_elem)
        else:
            ctx.save_for_backward(x_raw, w_raw)

        ctx.cfg = cfg
        ctx.has_bias = b is not None
        ctx.in_dim = w_raw.shape[1]
        ctx.out_dim = w_raw.shape[0]
        ctx.name = name

        # Compute linear (no bias)
        y = _F_linear(x, w)

        # Output quantization step 1 (post-matmul, pre-bias)
        out_schemes = cfg.output
        out_idx = 0
        if len(out_schemes) > 0:
            fp_y = y
            y = quantize(y, out_schemes[0], scale=output_scale)
            if emit_fn: emit_fn("output", out_idx, "output_post_quant", fp_y, y, out_schemes[0])
            out_idx += 1

        # Add bias + output quantization step 2 (post-bias-add)
        if q_bias is not None:
            y = y + q_bias
            if len(out_schemes) > 1:
                fp_y = y
                y = quantize(y, out_schemes[1], scale=output_scale)
                if emit_fn: emit_fn("output", out_idx, "output_post_quant", fp_y, y, out_schemes[1])
                out_idx += 1

        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        emit_fn = ctx.emit_fn
        out_dim = ctx.out_dim
        in_dim = ctx.in_dim

        # Quantize grad_output
        go_idx = 0
        for s in cfg.grad_output:
            fp_gy = grad_y
            grad_y = quantize(grad_y, s)
            if emit_fn: emit_fn("grad_output", go_idx, "grad_output_pre_quant", fp_gy, grad_y, s)
            go_idx += 1

        # --- grad_weight gemm ---
        # Saved tensors are post-elemwise (if training) or raw (if STE).
        # input_gw/grad_output_gw contain MX-only schemes.
        x_gw = x
        g_gw = grad_y
        for s in cfg.input_gw:
            x_gw = quantize(x_gw, s)
        for s in cfg.grad_output_gw:
            g_gw = quantize(g_gw, s)

        g_gw_2d = g_gw.reshape(-1, out_dim)
        x_gw_2d = x_gw.reshape(-1, in_dim)
        grad_w = g_gw_2d.T @ x_gw_2d

        gw_idx = 0
        for s in cfg.grad_weight:
            fp_gw = grad_w
            grad_w = quantize(grad_w, s)
            if emit_fn: emit_fn("grad_weight", gw_idx, "grad_weight_post_quant", fp_gw, grad_w, s)
            gw_idx += 1

        # --- grad_input gemm ---
        w_gi = w
        g_gi = grad_y
        for s in cfg.weight_gi:
            w_gi = quantize(w_gi, s)
        for s in cfg.grad_output_gi:
            g_gi = quantize(g_gi, s)

        grad_x = g_gi @ w_gi

        gi_idx = 0
        for s in cfg.grad_input:
            fp_gx = grad_x
            grad_x = quantize(grad_x, s)
            if emit_fn: emit_fn("grad_input", gi_idx, "grad_input_post_quant", fp_gx, grad_x, s)
            gi_idx += 1

        # --- grad_bias ---
        grad_b = None
        if ctx.has_bias:
            grad_b = grad_y.reshape(-1, out_dim).sum(0)
            for s in cfg.grad_bias:
                grad_b = quantize(grad_b, s)

        return grad_x, grad_w, grad_b, None, None, None, None

    @staticmethod
    def symbolic(g, x, w, b, cfg, name, emit_fn, output_scale=None):
        """ONNX symbolic: emit quantize nodes + MatMul + optional Add."""
        from src.onnx.helpers import _emit_quantize_node

        for scheme in cfg.input:
            x = _emit_quantize_node(g, x, scheme)

        for scheme in cfg.weight:
            w = _emit_quantize_node(g, w, scheme)

        wt = g.op("Transpose", w, perm_i=[1, 0])
        y = g.op("MatMul", x, wt)

        if len(cfg.output) > 0:
            y = _emit_quantize_node(g, y, cfg.output[0])

        if b is not None:
            qb = b
            for scheme in cfg.bias:
                qb = _emit_quantize_node(g, qb, scheme)
            y = g.op("Add", y, qb)

            if len(cfg.output) > 1:
                y = _emit_quantize_node(g, y, cfg.output[1])

        return y


class QuantizedLinear(ObservableMixin, nn.Linear):
    """Drop-in replacement for mx.Linear using OpQuantConfig.

    All quantization is driven by cfg — no mx_specs, no MxSpecs.
    ObservableMixin provides _emit for analysis (no-op in Phase 3).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        cfg: OpQuantConfig = None,
        name: str = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.cfg = cfg or OpQuantConfig()
        self._is_passthrough = self.cfg == OpQuantConfig()
        self._analysis_name = name

    def forward(self, x):
        if self._is_passthrough:
            return F.linear(x, self.weight, self.bias)

        emit_fn = self._emit if self._observers else None
        output_scale = self.get_buffer("_output_scale") \
            if hasattr(self, "_output_scale") else None
        return LinearFunction.apply(
            x, self.weight, self.bias, self.cfg, self._analysis_name, emit_fn,
            output_scale,
        )
