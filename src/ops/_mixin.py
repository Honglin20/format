"""
_QuantizedModuleMixin: shared cfg/inner_scheme initialization for Quantized* modules.
"""
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
