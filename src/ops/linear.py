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
    def forward(ctx, x, w, b, cfg: OpQuantConfig, name=None):
        # Save raw tensors for STE backward (when is_training=False)
        x_raw, w_raw = x, w

        # --- Input pipeline: elemwise first, then MX ---
        input_elem = tuple(s for s in cfg.input if s.granularity.mode != GranularityMode.PER_BLOCK)
        input_mx = tuple(s for s in cfg.input if s.granularity.mode == GranularityMode.PER_BLOCK)

        for s in input_elem:
            x = quantize(x, s)
        x_post_elem = x  # save intermediate for backward

        for s in input_mx:
            x = quantize(x, s)

        # --- Weight pipeline: elemwise first, then MX ---
        weight_elem = tuple(s for s in cfg.weight if s.granularity.mode != GranularityMode.PER_BLOCK)
        weight_mx = tuple(s for s in cfg.weight if s.granularity.mode == GranularityMode.PER_BLOCK)

        for s in weight_elem:
            w = quantize(w, s)
        w_post_elem = w

        for s in weight_mx:
            w = quantize(w, s)

        # --- Bias pipeline (elemwise only, no MX) ---
        q_bias = None
        if b is not None:
            q_bias = b
            for s in cfg.bias:
                q_bias = quantize(q_bias, s)

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
        y = F.linear(x, w)

        # Output quantization step 1 (post-matmul, pre-bias)
        out_schemes = cfg.output
        if len(out_schemes) > 0:
            y = quantize(y, out_schemes[0])

        # Add bias + output quantization step 2 (post-bias-add)
        if q_bias is not None:
            y = y + q_bias
            if len(out_schemes) > 1:
                y = quantize(y, out_schemes[1])

        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        out_dim = ctx.out_dim
        in_dim = ctx.in_dim

        # Quantize grad_output
        for s in cfg.grad_output:
            grad_y = quantize(grad_y, s)

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

        for s in cfg.grad_weight:
            grad_w = quantize(grad_w, s)

        # --- grad_input gemm ---
        w_gi = w
        g_gi = grad_y
        for s in cfg.weight_gi:
            w_gi = quantize(w_gi, s)
        for s in cfg.grad_output_gi:
            g_gi = quantize(g_gi, s)

        grad_x = g_gi @ w_gi

        for s in cfg.grad_input:
            grad_x = quantize(grad_x, s)

        # --- grad_bias ---
        grad_b = None
        if ctx.has_bias:
            grad_b = grad_y.reshape(-1, out_dim).sum(0)
            for s in cfg.grad_bias:
                grad_b = quantize(grad_b, s)

        return grad_x, grad_w, grad_b, None, None


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
        self._analysis_name = name

    def forward(self, x):
        if self.cfg == OpQuantConfig():
            return F.linear(x, self.weight, self.bias)

        return LinearFunction.apply(
            x, self.weight, self.bias, self.cfg, self._analysis_name,
        )
