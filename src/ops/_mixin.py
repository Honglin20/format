"""
_QuantizedModuleMixin: shared cfg/inner_scheme initialization for Quantized* modules.
"""
import torch

from src.scheme.op_config import OpQuantConfig


class _QuantizedModuleMixin:
    """Mixin providing shared cfg/inner_scheme validation and initialization.

    Usage in Quantized*.__init__:
        super().__init__(<module-specific args>)
        self._init_quant_cfg(cfg, inner_scheme, quantize_backprop, name)
    """

    def _init_quant_cfg(self, cfg, inner_scheme, quantize_backprop, name):
        if cfg is not None and inner_scheme is not None:
            raise ValueError("Cannot specify both cfg and inner_scheme")
        if inner_scheme is not None and cfg is None:
            bw = inner_scheme if quantize_backprop else None
            cfg = OpQuantConfig(input=inner_scheme, grad_input=bw)
        if cfg is None:
            cfg = OpQuantConfig()
        self.cfg = cfg
        self._analysis_name = name
        # quantize_nonlinear=True entry quantization — default to no-op
        self._entry_storage = None
        self._entry_compute = None

    def _entry_quantize(self, input):
        """Apply storage → compute entry quantization for quantize_nonlinear=True.

        When _entry_storage and _entry_compute are set (quantize_nonlinear=True
        with per_block compute), applies two-stage operand-entry quantization.
        Uses straight-through estimator (STE) so gradients flow through
        the quantization step.  Otherwise returns input unchanged (no-op).
        """
        from src.quantize import quantize
        if self._entry_storage is None and self._entry_compute is None:
            return input
        with torch.no_grad():
            q = input
            if self._entry_storage is not None:
                q = quantize(q, self._entry_storage)
            if self._entry_compute is not None:
                q = quantize(q, self._entry_compute)
        return input + (q - input).detach()
