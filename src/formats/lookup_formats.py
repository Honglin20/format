"""
Lookup-table-based quantization formats: NF4 from QLoRA.

These formats use a fixed set of discrete levels stored in a lookup table
rather than sign-magnitude float/integer representation.
quantize_elemwise() performs nearest-neighbor search against the LUT.
"""
import torch
from .base import FormatBase


class LookupFormat(FormatBase):
    """Base class for lookup-table-based quantization formats.

    quantization levels are a sorted 1D float32 tensor.
    quantize_elemwise() uses nearest-neighbor search.
    ebits=0 (no exponent), levels define the representable values.
    """

    __slots__ = ("levels",)

    def __init__(self, name: str, levels):
        self.name = name
        if not isinstance(levels, torch.Tensor):
            levels = torch.tensor(levels, dtype=torch.float32)
        if levels.ndim != 1:
            raise ValueError(
                f"levels must be a 1D tensor, got {levels.ndim}D"
            )
        self.levels = levels
        bit_count = levels.numel() - 1
        self.mbits = bit_count.bit_length() if bit_count > 0 else 1
        self.ebits = 0
        self.emax = 0
        self.max_norm = float(levels.abs().max().item())
        self.min_norm = 0.0
        self._freeze()

    def quantize(self, x, granularity, round_mode="nearest", allow_denorm=True,
                 scale=None):
        return super().quantize(x, granularity, round_mode, allow_denorm,
                                scale=scale)

    def quantize_elemwise(self, x, round_mode="nearest", allow_denorm=True,
                          saturate_normals=None):
        """Nearest-neighbor quantization against self.levels.

        Only round_mode='nearest' is supported. allow_denorm and
        saturate_normals are ignored (not applicable to LUT formats).
        """
        if round_mode != "nearest":
            raise ValueError(
                f"Lookup-based format {self.name!r} only supports "
                f"round_mode='nearest', got {round_mode!r}"
            )

        levels = self.levels.to(device=x.device, dtype=x.dtype)
        nan_mask = torch.isnan(x)

        x_safe = torch.where(nan_mask, torch.zeros_like(x), x)
        x_safe = torch.clamp(x_safe, -self.max_norm, self.max_norm)

        # Nearest-neighbor: x_safe shape (*), levels shape (L)
        # x_safe.unsqueeze(-1) → (*, 1) broadcast against (L,) → (*, L)
        distances = torch.abs(x_safe.unsqueeze(-1) - levels)
        indices = torch.argmin(distances, dim=-1)
        result = levels[indices]

        if nan_mask.any():
            result = result.clone()
            result[nan_mask] = float('nan')

        return result

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return (self.name == other.name
                and self.ebits == other.ebits
                and self.mbits == other.mbits
                and self.max_norm == other.max_norm
                and torch.equal(self.levels, other.levels))

    def __hash__(self):
        return hash(
            ("LookupFormat", self.name, self.ebits, self.mbits,
             tuple(self.levels.tolist()))
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}(name={self.name!r}, "
                f"mbits={self.mbits})")


class NF4Format(LookupFormat):
    """NF4 format from QLoRA (Dettmers et al., 2023).

    4-bit NormalFloat with 16 asymmetric quantization levels optimized
    for zero-centered normally distributed weights.
    Reference: https://arxiv.org/abs/2305.14314
    """

    __slots__ = ()

    NF4_LEVELS = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
        0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ]

    def __init__(self):
        super().__init__(name="nf4", levels=list(self.NF4_LEVELS))

    def __eq__(self, other):
        return isinstance(other, NF4Format)

    def __hash__(self):
        return hash("NF4Format")

    def __repr__(self):
        return "NF4Format()"
