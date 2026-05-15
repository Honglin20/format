"""
QuantizedLinear: OpQuantConfig-driven replacement for mx.Linear.

Forward + backward are bit-exact equivalent to mx/linear.py when driven by
the same OpQuantConfig produced by op_config_from_mx_specs.

Two-level quantization model (storage → compute):
- storage: applied first, always per-tensor elemwise (e.g. bfloat16)
- compute: per-role quantization (e.g. fp8 MX per-block)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scheme.op_config import OpQuantConfig
from src.quantize.elemwise import _enter_quantize, _exit_quantize

_F_linear = F.linear
from src.quantize import quantize
from src.observer.mixin import ObservableMixin


class LinearFunction(torch.autograd.Function):
    """Autograd function for quantized linear with QAT backward.

    Forward flow:
      1. storage quantize input (save post-storage for backward)
      2. compute quantize input
      3. storage quantize weight (save post-storage for backward)
      4. compute quantize weight
      5. storage quantize bias
      6. F.linear(qinput, qweight) — no bias
      7. storage quantize output[0] (post-matmul, pre-bias)
      8. add quantized bias
      9. storage quantize output[1] (post-bias-add)

    Backward flow:
      Saved tensors are post-storage (matching mx/linear.py's bf_in/bf_weight).
      Compute schemes in input_gw/grad_output_gw/weight_gi/grad_output_gi apply
      to already-storage-quantized tensors.
    """

    @staticmethod
    def forward(ctx, x, w, b, cfg: OpQuantConfig, name=None, emit_fn=None,
                output_scale=None, input_scale=None,
                output_mask=None, output_scale_o=None,
                input_mask=None, input_scale_o=None):
        ctx.emit_fn = emit_fn
        x_raw, w_raw = x, w

        # input: storage → compute
        # Use x_raw for ALL input observer fp32 references so QSNR measures
        # total input quantization quality (raw fp32 vs fully quantized).
        if cfg.storage is not None:
            x = quantize(x, cfg.storage)
            if emit_fn: emit_fn("input", 0, "input_pre_quant", x_raw, x, cfg.storage)
        x_post_storage = x
        if cfg.input is not None:
            x = quantize(x, cfg.input, scale=input_scale,
                         mask=input_mask, scale_o=input_scale_o)
            if emit_fn: emit_fn("input", 1, "input_pre_quant", x_raw, x, cfg.input)

        # weight: storage → compute
        # Use w_raw for ALL weight observer fp32 references.
        if cfg.storage is not None:
            w = quantize(w, cfg.storage)
            if emit_fn: emit_fn("weight", 0, "weight_pre_quant", w_raw, w, cfg.storage)
        w_post_storage = w
        if cfg.weight is not None:
            w = quantize(w, cfg.weight)
            if emit_fn: emit_fn("weight", 1, "weight_pre_quant", w_raw, w, cfg.weight)

        # bias: storage → compute
        q_bias = b
        if b is not None and cfg.storage is not None:
            fp_b = q_bias; q_bias = quantize(q_bias, cfg.storage)
            if emit_fn: emit_fn("bias", 0, "weight_pre_quant", fp_b, q_bias, cfg.storage)
        if b is not None and cfg.bias is not None:
            fp_b = q_bias; q_bias = quantize(q_bias, cfg.bias)
            if emit_fn: emit_fn("bias", 1, "weight_pre_quant", fp_b, q_bias, cfg.bias)

        # Save for backward: post-storage if training, raw if STE
        if cfg.is_training:
            ctx.save_for_backward(x_post_storage, w_post_storage)
        else:
            ctx.save_for_backward(x_raw, w_raw)

        ctx.cfg = cfg
        ctx.has_bias = b is not None
        ctx.in_dim = w_raw.shape[1]
        ctx.out_dim = w_raw.shape[0]
        ctx.name = name

        # matmul
        y = _F_linear(x, w)

        # Pre-compute true fp32 references for each intermediate output stage.
        # Stage 0 (post-matmul, pre-bias): compare against matmul-only true output.
        # Stages 1+ (post-bias onward): compare against full true output (incl. bias).
        #
        # Guard against QuantizeContext patches — torch.Tensor.__add__ is
        # patched inside QuantizeContext, which would quantize the bias
        # addition and corrupt the fp32 reference.
        _true_matmul = None
        _true_full = None
        if emit_fn is not None:
            _enter_quantize()
            try:
                _true_matmul = _F_linear(x_raw, w_raw)
                _true_full = _true_matmul + b if b is not None else _true_matmul
            finally:
                _exit_quantize()

        # output step 1 (post-matmul): storage (per-tensor, ignores scale)
        if cfg.storage is not None:
            y = quantize(y, cfg.storage)
            if emit_fn: emit_fn("output", 0, "output_post_quant", _true_matmul, y, cfg.storage)
        elif emit_fn is not None:
            _out_scheme = cfg.output or cfg.input or cfg.weight
            if _out_scheme is not None:
                emit_fn("output", 0, "output_post_quant", _true_matmul, y, _out_scheme)

        # bias add + output step 2 (post-bias): storage
        if q_bias is not None:
            _enter_quantize()
            try:
                y = y + q_bias
            finally:
                _exit_quantize()
            if cfg.storage is not None:
                y = quantize(y, cfg.storage)
                if emit_fn: emit_fn("output", 1, "output_post_quant", _true_full, y, cfg.storage)

        # output compute: calibrated scale applies here (per-channel / per-block)
        if cfg.output is not None:
            y = quantize(y, cfg.output, scale=output_scale,
                         mask=output_mask, scale_o=output_scale_o)
            if emit_fn: emit_fn("output", 2, "output_post_quant", _true_full, y, cfg.output)

        # Emit total layer error: true fp32 output vs final quantized output
        if emit_fn is not None:
            _out_scheme = cfg.output or cfg.weight or cfg.input or cfg.storage
            if _out_scheme is not None:
                emit_fn("output", 3, "layer_total", _true_full, y, _out_scheme)

        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        cfg: OpQuantConfig = ctx.cfg
        emit_fn = ctx.emit_fn
        out_dim = ctx.out_dim
        in_dim = ctx.in_dim

        # grad_output: storage → compute
        if cfg.storage is not None:
            fp_gy = grad_y; grad_y = quantize(grad_y, cfg.storage)
            if emit_fn: emit_fn("grad_output", 0, "grad_output_pre_quant", fp_gy, grad_y, cfg.storage)
        if cfg.grad_output is not None:
            fp_gy = grad_y; grad_y = quantize(grad_y, cfg.grad_output)
            if emit_fn: emit_fn("grad_output", 1, "grad_output_pre_quant", fp_gy, grad_y, cfg.grad_output)

        # grad_weight gemm
        x_gw = x
        if cfg.storage is not None:
            x_gw = quantize(x_gw, cfg.storage)
        if cfg.input_gw is not None:
            x_gw = quantize(x_gw, cfg.input_gw)

        g_gw = grad_y
        if cfg.storage is not None:
            g_gw = quantize(g_gw, cfg.storage)
        if cfg.grad_output_gw is not None:
            g_gw = quantize(g_gw, cfg.grad_output_gw)

        grad_w = g_gw.reshape(-1, out_dim).T @ x_gw.reshape(-1, in_dim)

        if cfg.storage is not None:
            fp_gw = grad_w; grad_w = quantize(grad_w, cfg.storage)
            if emit_fn: emit_fn("grad_weight", 0, "grad_weight_post_quant", fp_gw, grad_w, cfg.storage)
        if cfg.grad_weight is not None:
            fp_gw = grad_w; grad_w = quantize(grad_w, cfg.grad_weight)
            if emit_fn: emit_fn("grad_weight", 1, "grad_weight_post_quant", fp_gw, grad_w, cfg.grad_weight)

        # grad_input gemm
        w_gi = w
        if cfg.storage is not None:
            w_gi = quantize(w_gi, cfg.storage)
        if cfg.weight_gi is not None:
            w_gi = quantize(w_gi, cfg.weight_gi)

        g_gi = grad_y
        if cfg.storage is not None:
            g_gi = quantize(g_gi, cfg.storage)
        if cfg.grad_output_gi is not None:
            g_gi = quantize(g_gi, cfg.grad_output_gi)

        grad_x = g_gi @ w_gi

        if cfg.storage is not None:
            fp_gx = grad_x; grad_x = quantize(grad_x, cfg.storage)
            if emit_fn: emit_fn("grad_input", 0, "grad_input_post_quant", fp_gx, grad_x, cfg.storage)
        if cfg.grad_input is not None:
            fp_gx = grad_x; grad_x = quantize(grad_x, cfg.grad_input)
            if emit_fn: emit_fn("grad_input", 1, "grad_input_post_quant", fp_gx, grad_x, cfg.grad_input)

        # grad_bias
        grad_b = None
        if ctx.has_bias:
            grad_b = grad_y.reshape(-1, out_dim).sum(0)
            if cfg.storage is not None:
                grad_b = quantize(grad_b, cfg.storage)
            if cfg.grad_bias is not None:
                grad_b = quantize(grad_b, cfg.grad_bias)

        return grad_x, grad_w, grad_b, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, w, b, cfg, name, emit_fn, output_scale=None, input_scale=None,
                 output_mask=None, output_scale_o=None,
                 input_mask=None, input_scale_o=None):
        from src.onnx.helpers import _emit_quantize_node
        from src.session._context import _export_scales_var, _onnx_current_scale_var

        scales = _export_scales_var.get()
        current_scale = None
        if scales:
            current_scale = scales.get(name)
            if current_scale is None:
                current_scale = scales.get("")
        _onnx_current_scale_var.set(current_scale)
        try:
            if cfg.storage is not None:
                x = _emit_quantize_node(g, x, cfg.storage)
            if cfg.input is not None:
                x = _emit_quantize_node(g, x, cfg.input)

            if cfg.storage is not None:
                w = _emit_quantize_node(g, w, cfg.storage)
            if cfg.weight is not None:
                w = _emit_quantize_node(g, w, cfg.weight)

            wt = g.op("Transpose", w, perm_i=[1, 0])
            y = g.op("MatMul", x, wt)

            if cfg.storage is not None:
                y = _emit_quantize_node(g, y, cfg.storage)

            if b is not None:
                if cfg.storage is not None:
                    b = _emit_quantize_node(g, b, cfg.storage)
                if cfg.bias is not None:
                    b = _emit_quantize_node(g, b, cfg.bias)
                y = g.op("Add", y, b)
                if cfg.storage is not None:
                    y = _emit_quantize_node(g, y, cfg.storage)

            if cfg.output is not None:
                y = _emit_quantize_node(g, y, cfg.output)
            return y
        finally:
            _onnx_current_scale_var.set(None)


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
        input_scale = self.get_buffer("_input_scale") \
            if hasattr(self, "_input_scale") else None
        output_mask = self.get_buffer("_output_mask") \
            if hasattr(self, "_output_mask") else None
        output_scale_o = self.get_buffer("_output_scale_o") \
            if hasattr(self, "_output_scale_o") else None
        input_mask = self.get_buffer("_input_mask") \
            if hasattr(self, "_input_mask") else None
        input_scale_o = self.get_buffer("_input_scale_o") \
            if hasattr(self, "_input_scale_o") else None
        return LinearFunction.apply(
            x, self.weight, self.bias, self.cfg, self._analysis_name, emit_fn,
            output_scale, input_scale,
            output_mask, output_scale_o,
            input_mask, input_scale_o,
        )
