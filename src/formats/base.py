"""
FormatBase: Abstract base class for all quantization formats.

Replaces the old ElemFormat enum + _get_format_params() if-elif chain
with extensible strategy objects. Instances are immutable after construction.
"""
from abc import ABC, abstractmethod

import torch

_VALID_ROUND_MODES = {"nearest", "floor", "even", "dither"}


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

    As a frozen-dataclass field in QuantScheme, instances must support
    value-based equality and hashing — subclasses must implement __eq__/__hash__.
    """

    __slots__ = ("name", "ebits", "mbits", "emax", "max_norm", "min_norm", "_frozen")

    @abstractmethod
    def __eq__(self, other) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

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

    def quantize_elemwise(self, x, round_mode="nearest", allow_denorm=True,
                          saturate_normals=None):
        """Element-wise quantization — the format's atomic quantization operation.

        Maps values in the representable range to the format's discrete levels.
        This is the unified, mandatory step that EVERY quantization pipeline
        (per-tensor, per-channel, per-block) must go through.

        Subclasses with non-standard quantization algorithms (e.g., NF4 with
        a lookup table) should override this method. The default implementation
        uses sign-magnitude representation with ebits/mbits/max_norm.

        Args:
            x: Input tensor, assumed scaled to the format's representable range.
            round_mode: "nearest" | "floor" | "even" | "dither"
            allow_denorm: If False, flush subnormal values to zero.
            saturate_normals: If None, defaults to True for integer formats
                (ebits==0) and False for float formats. If True, values
                exceeding max_norm are clamped; if False, set to ±Inf.

        Returns:
            Quantized tensor with same shape as x.
        """
        from src.quantize.elemwise import _quantize_elemwise_core
        if saturate_normals is None:
            saturate_normals = (self.ebits == 0)
        return _quantize_elemwise_core(
            x, self.mbits, self.ebits, self.max_norm,
            round_mode=round_mode, allow_denorm=allow_denorm,
            saturate_normals=saturate_normals,
        )

    @abstractmethod
    def quantize(self, x, granularity, round_mode="nearest", allow_denorm=True):
        """Quantize tensor x to this format.

        Declared @abstractmethod to force subclasses to explicitly decide
        whether to use this default dispatch or override with special logic.
        Subclasses can call super().quantize() to reuse the default.

        Args:
            x: Input tensor.
            granularity: GranularitySpec controlling scale sharing.
            round_mode: "nearest" | "floor" | "even" | "dither"
            allow_denorm: If False, flush subnormal values to zero (float formats only).

        Returns:
            Quantized tensor with same shape as x.
        """
        if round_mode not in _VALID_ROUND_MODES:
            raise ValueError(
                f"Invalid round_mode {round_mode!r}. Must be one of {_VALID_ROUND_MODES}"
            )
        from src.scheme.granularity import GranularityMode
        mode = granularity.mode
        if mode == GranularityMode.PER_TENSOR:
            return self._quantize_per_tensor(x, round_mode, allow_denorm)
        elif mode == GranularityMode.PER_CHANNEL:
            return self._quantize_per_channel(x, granularity, round_mode, allow_denorm)
        elif mode == GranularityMode.PER_BLOCK:
            return self._quantize_per_block(x, granularity, round_mode)
        raise ValueError(f"Unknown granularity mode: {mode}")

    def _quantize_per_tensor(self, x, round_mode, allow_denorm=True):
        """Default per-tensor quantization: element-wise only (no scaling)."""
        return self.quantize_elemwise(x, round_mode=round_mode,
                                      allow_denorm=allow_denorm)

    def _quantize_per_channel(self, x, granularity, round_mode, allow_denorm=True):
        """Default per-channel quantization: compute per-channel scale, then elemwise."""
        axis = granularity.channel_axis
        if axis < 0:
            axis = x.ndim + axis
        if not (0 <= axis < x.ndim):
            raise ValueError(
                f"channel_axis={granularity.channel_axis} out of range "
                f"for tensor with ndim={x.ndim}"
            )

        amax = torch.amax(torch.abs(x), dim=axis, keepdim=True)
        amax = amax.clamp(min=1e-12)

        # Normalize to [-1, 1], quantize, then rescale
        x_norm = x / amax
        x_q = self.quantize_elemwise(x_norm, round_mode=round_mode,
                                     allow_denorm=allow_denorm)
        return x_q * amax

    def _quantize_per_block(self, x, granularity, round_mode):
        """Default per-block quantization: delegate to _quantize_mx.

        During JIT tracing (ONNX export), return x unchanged — the Function's
        symbolic() method handles quantization in the ONNX graph.  The traced
        forward() is only used for shape inference.
        """
        if torch.jit.is_tracing():
            return x
        from src.quantize.mx_quantize import _quantize_mx
        return _quantize_mx(
            x, scale_bits=8, elem_format=self,
            block_size=granularity.block_size,
            axes=granularity.block_axis, round_mode=round_mode,
        )

    def export_onnx(self, g, x, scheme):
        """Emit ONNX nodes for this format's quantize step.

        Default: emit as com.microxscaling::MxQuantize custom node.
        Subclasses that have standard ONNX representations (e.g. IntFormat
        for QDQ, standard FP8 for QDQ) override this.

        Args:
            g: TorchScript ONNX graph builder.
            x: Input graph value.
            scheme: The full QuantScheme (format + granularity + transform).

        Returns:
            ONNX graph value representing quantized-then-dequantized x.
        """
        from src.scheme.granularity import GranularityMode
        block_size = (scheme.granularity.block_size
                      if scheme.granularity.mode == GranularityMode.PER_BLOCK
                      else 0)
        return g.op(
            "com.microxscaling::MxQuantize",
            x,
            elem_format_s=self.name,
            block_size_i=block_size,
            round_mode_s=scheme.round_mode,
        )

    @staticmethod
    def from_str(s: str) -> "FormatBase":
        """Factory: look up format by string name in the registry."""
        from .registry import get_format
        return get_format(s)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, ebits={self.ebits}, mbits={self.mbits})"
