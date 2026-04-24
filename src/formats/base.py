"""
FormatBase: Abstract base class for all quantization formats.

Replaces the old ElemFormat enum + _get_format_params() if-elif chain
with extensible strategy objects. Instances are immutable after construction.
"""
from abc import ABC


def compute_min_norm(ebits: int) -> float:
    """Compute min normal number. Returns 0 for integer formats (ebits==0)."""
    if ebits == 0:
        return 0.0
    emin = 2 - (2 ** (ebits - 1))
    return 2 ** emin


def compute_max_norm(ebits: int, mbits: int) -> float:
    """Compute max normal number for float formats that define NaN/Inf (ebits >= 5)."""
    emax = 2 ** (ebits - 1) - 1
    return 2 ** emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)


class FormatBase(ABC):
    """Abstract base for all quantization formats.

    Subclasses must set: name, ebits, mbits, emax, max_norm, min_norm
    in their __init__. After __init__, these attributes are frozen (immutable).
    """

    __slots__ = ("name", "ebits", "mbits", "emax", "max_norm", "min_norm", "_frozen")

    def _freeze(self):
        """Call at end of subclass __init__ to make instance immutable."""
        object.__setattr__(self, "_frozen", True)

    @staticmethod
    def _all_slots(cls):
        """Collect __slots__ across the entire MRO."""
        slots = set()
        for klass in cls.__mro__:
            slots.update(getattr(klass, '__slots__', ()))
        return slots

    def __setattr__(self, key, value):
        # Reject attributes not in __slots__ even before freeze
        if key != "_frozen" and key not in FormatBase._all_slots(self.__class__):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {key!r}"
            )
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{self.__class__.__name__} is immutable after construction"
            )
        object.__setattr__(self, key, value)

    @property
    def is_integer(self) -> bool:
        return self.ebits == 0

    @staticmethod
    def from_str(s: str) -> "FormatBase":
        """Factory: look up format by string name in the registry."""
        from .registry import get_format
        return get_format(s)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, ebits={self.ebits}, mbits={self.mbits})"
