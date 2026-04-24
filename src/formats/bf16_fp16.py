"""
Standard IEEE float formats: bfloat16, float16.
"""
from .base import FormatBase, compute_min_norm, compute_max_norm


class BFloat16Format(FormatBase):
    """Bfloat16: 1 sign + 8 exp + 7 mantissa (8+1=9 mbits with implicit)."""

    __slots__ = ()

    def __init__(self):
        self.name = "bfloat16"
        self.ebits = 8
        self.mbits = 9
        self.emax = 2 ** (self.ebits - 1) - 1  # 127
        self.max_norm = compute_max_norm(self.ebits, self.mbits)
        self.min_norm = compute_min_norm(self.ebits)
        self._freeze()


class Float16Format(FormatBase):
    """IEEE float16: 1 sign + 5 exp + 10 mantissa (5+12=12 mbits with implicit)."""

    __slots__ = ()

    def __init__(self):
        self.name = "float16"
        self.ebits = 5
        self.mbits = 12
        self.emax = 2 ** (self.ebits - 1) - 1  # 15
        self.max_norm = compute_max_norm(self.ebits, self.mbits)
        self.min_norm = compute_min_norm(self.ebits)
        self._freeze()
