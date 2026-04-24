"""
Standard IEEE float formats: bfloat16, float16.
"""
import torch
from .base import FormatBase, compute_min_norm, compute_max_norm, _VALID_ROUND_MODES
from src.scheme.granularity import GranularityMode


class BFloat16Format(FormatBase):
    """Bfloat16: 1 sign + 8 exp + 7 mantissa (8+1=9 mbits with implicit).

    round_mode='even'+per_tensor uses the hardware .to(torch.bfloat16) shortcut.
    CUDA bf16 availability check is deferred to the operator layer where device
    context is known; the shortcut always works on CPU and on CUDA GPUs that
    support bf16 (which is virtually all current hardware).
    """

    __slots__ = ()

    def __init__(self):
        self.name = "bfloat16"
        self.ebits = 8
        self.mbits = 9
        self.emax = 2 ** (self.ebits - 1) - 1  # 127
        self.max_norm = compute_max_norm(self.ebits, self.mbits)
        self.min_norm = compute_min_norm(self.ebits)
        self._freeze()

    def __eq__(self, other):
        return isinstance(other, BFloat16Format)

    def __hash__(self):
        return hash("BFloat16Format")

    def quantize(self, x, granularity, round_mode="nearest"):
        if round_mode not in _VALID_ROUND_MODES:
            raise ValueError(
                f"Invalid round_mode {round_mode!r}. Must be one of {_VALID_ROUND_MODES}"
            )
        if round_mode == "even" and granularity.mode == GranularityMode.PER_TENSOR:
            return x.to(torch.bfloat16).float()
        return super().quantize(x, granularity, round_mode)


class Float16Format(FormatBase):
    """IEEE float16: 1 sign + 5 exp + 10 mantissa (5+12=12 mbits with implicit).

    round_mode='even'+per_tensor uses the hardware .to(torch.float16) shortcut.
    """

    __slots__ = ()

    def __init__(self):
        self.name = "float16"
        self.ebits = 5
        self.mbits = 12
        self.emax = 2 ** (self.ebits - 1) - 1  # 15
        self.max_norm = compute_max_norm(self.ebits, self.mbits)
        self.min_norm = compute_min_norm(self.ebits)
        self._freeze()

    def __eq__(self, other):
        return isinstance(other, Float16Format)

    def __hash__(self):
        return hash("Float16Format")

    def quantize(self, x, granularity, round_mode="nearest"):
        if round_mode not in _VALID_ROUND_MODES:
            raise ValueError(
                f"Invalid round_mode {round_mode!r}. Must be one of {_VALID_ROUND_MODES}"
            )
        if round_mode == "even" and granularity.mode == GranularityMode.PER_TENSOR:
            return x.to(torch.float16).float()
        return super().quantize(x, granularity, round_mode)
