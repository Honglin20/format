"""
Integer quantization formats: int2, int4, int8.

Sign-magnitude representation with 1.xxx implicit format.
"""
from .base import FormatBase, compute_min_norm


class IntFormat(FormatBase):
    """Integer format: int2, int4, int8.

    Represented as sign-magnitude with mbits total bits.
    ebits=0, emax=0, max_norm uses same formula as old code.
    """

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
