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

    __slots__ = ("name", "ebits", "mbits", "emax", "max_norm", "min_norm", "_frozen", "_hardware_dtype")

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
        from src.formats._core import _elemwise_core
        if saturate_normals is None:
            saturate_normals = (self.ebits == 0)
        return _elemwise_core(
            x, self.mbits, self.ebits, self.max_norm,
            round_mode=round_mode, allow_denorm=allow_denorm,
            saturate_normals=saturate_normals,
        )

    def quantize(self, x, granularity, round_mode="nearest", allow_denorm=True,
                 scale=None, scale_storage="pot"):
        """Quantize tensor x to this format.

        Dispatches by granularity mode.  Subclasses may override to provide
        hardware shortcuts or specialized dispatch logic.

        Args:
            x: Input tensor.
            granularity: GranularitySpec controlling scale sharing.
            round_mode: "nearest" | "floor" | "even" | "dither"
            allow_denorm: If False, flush subnormal values to zero (float formats only).
            scale: Optional pre-computed scale tensor.  If provided, skips
                on-the-fly scale computation and uses this directly.
            scale_storage: "pot" (default) or "fp32".  When "pot", the amax
                is rounded to the nearest power of 2 before normalization.
                Per_block is inherently POT (MX shared exponents); scale_storage
                has no effect there.

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
            return self._quantize_per_tensor(x, round_mode, allow_denorm, scale=scale,
                                              scale_storage=scale_storage)
        elif mode == GranularityMode.PER_CHANNEL:
            return self._quantize_per_channel(x, granularity, round_mode, allow_denorm,
                                              scale=scale, scale_storage=scale_storage)
        elif mode == GranularityMode.PER_BLOCK:
            return self._quantize_per_block(x, granularity, round_mode,
                                              scale=scale, scale_storage=scale_storage)
        raise ValueError(f"Unknown granularity mode: {mode}")

    def _quantize_per_tensor(self, x, round_mode, allow_denorm=True, scale=None,
                              scale_storage="pot"):
        """Default per-tensor quantization.

        When ``_hardware_dtype`` is set and the preconditions are met
        (round_mode='even', allow_denorm=True), uses the hardware dtype
        conversion shortcut for formats like bfloat16/float16.

        Float formats (ebits > 0) are quantized directly — their dynamic
        range already covers real-world tensor values, matching mx/ behaviour.

        Integer formats (ebits == 0, e.g. int8/int4/int2) are normalised to
        [-1, 1] before elemwise quantisation because their max_norm (~1–2)
        is too small to represent raw tensor values without clamping.

        When ``scale_storage="pot"`` (default), the scalar amax is rounded
        to the nearest power of 2 before normalization.
        """
        hw_dtype = getattr(self, "_hardware_dtype", None)
        if (hw_dtype is not None
                and round_mode == "even"
                and allow_denorm):
            return x.to(hw_dtype).float()

        # Float formats: direct elemwise (matching mx/ behaviour).
        # Their exponent range covers practical tensor values without scaling.
        if self.ebits > 0:
            return self.quantize_elemwise(x, round_mode=round_mode,
                                          allow_denorm=allow_denorm)

        # Integer formats: normalise → elemwise → rescale.
        if scale is not None:
            amax = scale
        else:
            amax = torch.amax(torch.abs(x))

        if not torch.isfinite(amax) or amax <= 0:
            return self.quantize_elemwise(x, round_mode=round_mode,
                                          allow_denorm=allow_denorm)

        amax = amax.clamp(min=1e-12)
        if scale_storage == "pot":
            amax = 2 ** torch.round(torch.log2(amax))
        x_norm = x / amax
        x_q = self.quantize_elemwise(x_norm, round_mode=round_mode,
                                      allow_denorm=allow_denorm)
        return x_q * amax

    def _quantize_per_channel(self, x, granularity, round_mode, allow_denorm=True,
                              scale=None, scale_storage="pot"):
        """Default per-channel quantization: compute per-channel scale, then elemwise.

        If ``scale`` is provided, it is used directly as ``amax``, skipping
        the on-the-fly ``torch.amax(torch.abs(x))`` computation.

        When ``scale_storage="pot"``, the amax is rounded to the nearest power
        of 2 before normalization.
        """
        if scale is not None:
            amax = scale
        else:
            axis = granularity.channel_axis
            if axis < 0:
                axis = x.ndim + axis
            if not (0 <= axis < x.ndim):
                raise ValueError(
                    f"channel_axis={granularity.channel_axis} out of range "
                    f"for tensor with ndim={x.ndim}"
                )
            # Reduce all dims EXCEPT channel_axis to get per-channel amax.
            dims_to_reduce = [i for i in range(x.ndim) if i != axis]
            amax = torch.amax(torch.abs(x), dim=tuple(dims_to_reduce), keepdim=True)
            amax = amax.clamp(min=1e-12)

        if scale_storage == "pot":
            amax = 2 ** torch.round(torch.log2(amax))

        # Normalize to [-1, 1], quantize, then rescale
        x_norm = x / amax
        x_q = self.quantize_elemwise(x_norm, round_mode=round_mode,
                                     allow_denorm=allow_denorm)
        return x_q * amax

    def _quantize_per_block(self, x, granularity, round_mode, scale=None,
                              scale_storage="pot",
                              _shared_exp_method="max",
                              _flush_fp32_subnorms=False):
        """Per-block quantization with MX-style shared exponents.

        Same structure as _quantize_per_channel:
        tile into blocks → compute shared exponent → normalize →
        elemwise quantize → rescale → until back to original shape.

        MX shared exponents are inherently power-of-two; ``scale_storage``
        has no effect in this path.

        During JIT tracing (ONNX export), return x unchanged — the
        Function's symbolic() method handles quantization in the ONNX graph.
        """
        if torch.jit.is_tracing():
            return x

        if granularity.outlier_ratio > 0.0:
            from src.formats._outlier_utils import _quantize_outlier_bank
            return _quantize_outlier_bank(
                self, x, granularity, round_mode, scale_storage=scale_storage)

        from src.formats._block_utils import (
            _reshape_to_blocks,
            _undo_reshape_to_blocks,
            _shared_exponents,
            FP32_EXPONENT_BIAS,
        )

        block_size = granularity.block_size
        axes = [granularity.block_axis]

        # Step 1: normalize axes to non-negative
        axes = [a + x.ndim if a < 0 else a for a in axes]

        # Step 2: tile into hardware-vector-sized blocks
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            x, axes, block_size)

        # Step 3: compute shared exponents per block
        shared_exp_axes = [a + 1 for a in axes]
        shared_exp = _shared_exponents(
            A, method=_shared_exp_method, axes=shared_exp_axes, ebits=0)

        # Step 3b: flush subnormal FP32 inputs to zero (legacy mx path)
        if _flush_fp32_subnorms:
            A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

        # Step 4: offset by format's max representable exponent
        shared_exp = shared_exp - self.emax

        # Step 5: clamp shared exponents to int8 range (scale_bits=8)
        scale_emax = 2**(8-1) - 1
        shared_exp[shared_exp > scale_emax] = float("NaN")
        shared_exp[shared_exp < -scale_emax] = -scale_emax

        # Step 6: normalize by shared exponent
        A = A / (2 ** shared_exp)

        # Step 7: element-wise quantize to target format
        A = self.quantize_elemwise(A, round_mode=round_mode,
                                   allow_denorm=True, saturate_normals=True)

        # Step 8: rescale
        A = A * (2 ** shared_exp)

        # Step 9: undo block tiling
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

        return A

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
