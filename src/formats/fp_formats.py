"""
Float quantization formats: fp8_e5m2, fp8_e4m3, fp6_e3m2, fp6_e2m3, fp4_e2m1.

MX sub-byte floating point formats with various exponent/mantissa distributions.
"""
from .base import FormatBase, compute_min_norm, compute_max_norm


class FPFormat(FormatBase):
    """MX sub-byte floating point format.

    mbits includes sign bit and implicit one bit (same convention as old code).
    """

    __slots__ = ()

    def __init__(self, name: str, ebits: int, mbits: int, max_norm_override: float = None):
        self.name = name
        self.ebits = ebits
        self.mbits = mbits

        # Compute emax
        if ebits >= 5:
            # Standard float with NaN/Inf
            self.emax = 2 ** (ebits - 1) - 1
        elif ebits > 0:
            # Sub-byte float without NaN/Inf
            self.emax = 2 ** (ebits - 1)
        else:
            self.emax = 0

        # Compute max_norm
        if max_norm_override is not None:
            self.max_norm = max_norm_override
        elif ebits >= 5:
            self.max_norm = compute_max_norm(ebits, mbits)
        else:
            # Sub-byte float: use the sub-byte emax formula
            self.max_norm = 2 ** self.emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)

        self.min_norm = compute_min_norm(ebits)
        self._freeze()

    def __eq__(self, other):
        return (isinstance(other, FPFormat)
                and self.name == other.name
                and self.ebits == other.ebits
                and self.mbits == other.mbits
                and self.max_norm == other.max_norm)

    def __hash__(self):
        return hash(("FPFormat", self.name, self.ebits, self.mbits))

    def quantize(self, x, granularity, round_mode="nearest"):
        return super().quantize(x, granularity, round_mode)
