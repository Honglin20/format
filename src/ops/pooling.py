"""
Quantized AdaptiveAvgPool2d operator — inner_scheme-driven, bit-exact equivalent to mx/adaptive_avg_pooling.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.analysis.mixin import ObservableMixin
from src.ops.vec_ops import vec_add, vec_reduce_mean

_f_adaptive_avg_pool2d = F.adaptive_avg_pool2d


def _start_index(a, b, c):
    return math.floor((float(a) * float(c)) / b)


def _end_index(a, b, c):
    return math.ceil((float(a + 1) * float(c)) / b)


class AdaptiveAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output_size, inner_scheme, quantize_backprop=True, name=None, emit_fn=None):
        ctx.name = name
        ctx.emit_fn = emit_fn

        sizeB, sizeD, isizeH, isizeW = input.size()

        if isinstance(output_size, tuple) and len(output_size) == 2:
            osizeH = output_size[0] if output_size[0] else isizeH
            osizeW = output_size[1] if output_size[1] else isizeW
        elif isinstance(output_size, int):
            osizeH, osizeW = output_size, output_size
        elif output_size is None:
            osizeH, osizeW = isizeH, isizeW
        else:
            raise ValueError(
                f'expected 1D or 2D output_size (got {len(output_size)}D output_size)')

        if input.dim() != 4:
            raise ValueError(
                f'expected 4D input (got {input.dim()}D input)')

        device = input.device
        output = torch.zeros(sizeB, sizeD, osizeH, osizeW, device=device)

        for oh in range(osizeH):
            istartH = _start_index(oh, osizeH, isizeH)
            iendH = _end_index(oh, osizeH, isizeH)

            for ow in range(osizeW):
                istartW = _start_index(ow, osizeW, isizeW)
                iendW = _end_index(ow, osizeW, isizeW)

                input_slice = input[:, :, istartH:iendH, istartW:iendW]
                output[:, :, oh, ow] = vec_reduce_mean(
                    input_slice, [2, 3], keepdim=False, scheme=inner_scheme)

        ctx.osizeH = osizeH
        ctx.osizeW = osizeW
        ctx.sizeB = sizeB
        ctx.sizeD = sizeD
        ctx.isizeH = isizeH
        ctx.isizeW = isizeW
        ctx.device = device
        ctx.inner_scheme_bw = inner_scheme if quantize_backprop else None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        osizeH, osizeW = ctx.osizeH, ctx.osizeW
        sizeB, sizeD = ctx.sizeB, ctx.sizeD
        isizeH, isizeW = ctx.isizeH, ctx.isizeW
        device = ctx.device
        scheme = ctx.inner_scheme_bw

        grad_input = torch.zeros(sizeB, sizeD, isizeH, isizeW, device=device)

        for oh in range(osizeH):
            istartH = _start_index(oh, osizeH, isizeH)
            iendH = _end_index(oh, osizeH, isizeH)
            kH = iendH - istartH

            for ow in range(osizeW):
                istartW = _start_index(ow, osizeW, isizeW)
                iendW = _end_index(ow, osizeW, isizeW)
                kW = iendW - istartW

                grad_delta = grad_output[:, :, oh, ow] / kH / kW

                target_shape = [sizeB, sizeD, kH, kW]
                expanded_grad_delta = grad_delta.view(
                    *grad_delta.shape,
                    *(1,) * (len(target_shape) - grad_delta.ndim)
                ).expand(target_shape)

                grad_input[:, :, istartH:iendH, istartW:iendW] = vec_add(
                    grad_input[:, :, istartH:iendH, istartW:iendW],
                    expanded_grad_delta,
                    scheme,
                )

        return (grad_input, None, None, None, None, None)


class QuantizedAdaptiveAvgPool2d(ObservableMixin, nn.Module):
    def __init__(self, output_size, cfg: OpQuantConfig = None,
                 inner_scheme: QuantScheme = None,
                 quantize_backprop: bool = True, name: str = None, **kwargs):
        super().__init__()
        self.output_size = output_size
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            fwd_pipeline = (inner_scheme,)
            bw_pipeline = (inner_scheme,) if quantize_backprop else ()
            cfg = OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name

    def forward(self, input):
        inner_scheme = self.cfg.input[0] if self.cfg.input else None
        quantize_backprop = bool(self.cfg.grad_input)
        if inner_scheme is None:
            return _f_adaptive_avg_pool2d(input, self.output_size)
        emit_fn = self._emit if self._observers else None
        return AdaptiveAvgPool2dFunction.apply(
            input, self.output_size, inner_scheme,
            quantize_backprop, self._analysis_name, emit_fn,
        )
