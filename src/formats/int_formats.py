"""
Integer quantization formats: int2, int4, int8.

Sign-magnitude representation with 1.xxx implicit format.
"""
import torch
from .base import FormatBase, compute_min_norm


class IntFormat(FormatBase):
    """Integer format: int2, int4, int8.

    Represented as sign-magnitude with mbits total bits.
    ebits=0, emax=0, max_norm uses same formula as old code.
    """

    __slots__ = ()

    def __init__(self, bits: int, name: str = None):
        self.name = name or f"int{bits}"
        self.ebits = 0
        self.mbits = bits
        self.emax = 0
        # Same formula as old code: 2**emax * float(2**(mbits-1) - 1) / 2**(mbits-2)
        # With emax=0: 1 * (2**(mbits-1) - 1) / 2**(mbits-2)
        self.max_norm = float(2 ** (self.mbits - 1) - 1) / 2 ** (self.mbits - 2)
        self.min_norm = compute_min_norm(self.ebits)
        self._freeze()

    def __eq__(self, other):
        return isinstance(other, IntFormat) and self.name == other.name and self.mbits == other.mbits

    def __hash__(self):
        return hash(("IntFormat", self.name, self.mbits))

    def export_onnx(self, g, x, scheme):
        from src.scheme.granularity import GranularityMode
        if scheme.granularity.mode == GranularityMode.PER_BLOCK:
            return super().export_onnx(g, x, scheme)
        scale = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
        zp = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
        xq = g.op("QuantizeLinear", x, scale, zp)
        return g.op("DequantizeLinear", xq, scale, zp)

    def quantize(self, x, granularity, round_mode="nearest", allow_denorm=True):
        return super().quantize(x, granularity, round_mode, allow_denorm)
