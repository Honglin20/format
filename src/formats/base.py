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
                 scale=None, scale_storage="pot", mask=None, scale_o=None,
                 outlier_format=None):
        """Quantize tensor x to this format.

        Dispatches by granularity mode.  Subclasses may override to provide
        hardware shortcuts or specialized dispatch logic.

        Args:
            x: Input tensor.
            granularity: GranularitySpec controlling scale sharing.
            round_mode: "nearest" | "floor" | "even" | "dither"
            allow_denorm: If False, flush subnormal values to zero (float formats only).
            scale: Optional pre-computed scale tensor (normal-group amax when sparse).
            scale_storage: "pot" (default) or "fp32".
            mask: Optional pre-computed boolean mask for static sparse.
                  True = outlier.  Requires scale and scale_o.
            scale_o: Optional pre-computed scale for outlier group (static sparse).
            outlier_format: If set, outlier group uses this format instead of self.

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
            if granularity.outlier_ratio > 0.0:
                if mask is not None:
                    return self._quantize_static_sparse(
                        x, mask, scale, scale_o, round_mode, allow_denorm,
                        scale_storage, outlier_format=outlier_format)
                # Dynamic sparse — scale is ignored (recomputed via topk).
                # Calibrated scales pass through here when modules have
                # _output_scale buffers but no static sparse mask yet.
                return self._quantize_per_tensor_sparse(
                    x, granularity, round_mode, allow_denorm,
                    scale_storage=scale_storage, outlier_format=outlier_format)
            return self._quantize_per_tensor(x, round_mode, allow_denorm, scale=scale,
                                              scale_storage=scale_storage)
        elif mode == GranularityMode.PER_CHANNEL:
            if granularity.outlier_ratio > 0.0:
                if mask is not None:
                    return self._quantize_static_sparse(
                        x, mask, scale, scale_o, round_mode, allow_denorm,
                        scale_storage, outlier_format=outlier_format)
                return self._quantize_per_channel_sparse(
                    x, granularity, round_mode, allow_denorm,
                    scale_storage=scale_storage, outlier_format=outlier_format)
            return self._quantize_per_channel(x, granularity, round_mode, allow_denorm,
                                              scale=scale, scale_storage=scale_storage)
        elif mode == GranularityMode.PER_BLOCK:
            if granularity.outlier_ratio > 0.0 and mask is not None:
                raise NotImplementedError(
                    "PER_BLOCK static sparse is not yet implemented. "
                    "Use dynamic sparse (mask=None) or per_tensor/per_channel/bank "
                    "granularity for static sparse."
                )
            return self._quantize_per_block(x, granularity, round_mode,
                                              scale=scale, scale_storage=scale_storage,
                                              outlier_format=outlier_format)
        elif mode == GranularityMode.BANK:
            if granularity.outlier_ratio > 0.0:
                if mask is not None:
                    return self._quantize_per_bank_static_sparse(
                        x, mask, scale, scale_o, granularity, round_mode,
                        allow_denorm=allow_denorm, scale_storage=scale_storage,
                        outlier_format=outlier_format)
                return self._quantize_per_bank_sparse(
                    x, granularity, round_mode, allow_denorm=allow_denorm,
                    scale_storage=scale_storage, outlier_format=outlier_format)
            return self._quantize_per_bank(x, granularity, round_mode,
                                            allow_denorm=allow_denorm,
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
        if torch.jit.is_tracing():
            return x
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
        if torch.jit.is_tracing():
            return x
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

    def _quantize_per_bank(self, x, granularity, round_mode, allow_denorm=True,
                           scale=None, scale_storage="pot"):
        """Per-bank quantization: split along bank_axis into banks.

        Each bank spans ALL elements across non-bank dimensions within its
        bank_axis segment — unlike PER_BLOCK which subdivides every dimension.
        One amax per bank. Supports fp32 and pot scale_storage.
        """
        if torch.jit.is_tracing():
            return x
        axis = granularity.bank_axis
        if axis < 0:
            axis = x.ndim + axis
        if not (0 <= axis < x.ndim):
            raise ValueError(
                f"bank_axis={granularity.bank_axis} out of range "
                f"for tensor with ndim={x.ndim}"
            )

        bank_size = granularity.bank_size
        N_along = x.shape[axis]
        if N_along % bank_size != 0:
            raise ValueError(
                f"Dimension {axis} size {N_along} not divisible "
                f"by bank_size {bank_size}"
            )

        num_banks = N_along // bank_size

        # Reshape: split axis into (num_banks, bank_size)
        new_shape = list(x.shape)
        new_shape[axis] = num_banks
        new_shape.insert(axis + 1, bank_size)
        x_r = x.reshape(new_shape)
        # x_r shape: (..., num_banks, bank_size, ...)
        # bank dim is at position `axis`, inner dim at `axis+1`

        if scale is not None:
            amax = scale
        else:
            # Reduce all dims EXCEPT the bank dim
            dims_to_reduce = [i for i in range(x_r.ndim) if i != axis]
            amax = torch.amax(torch.abs(x_r), dim=tuple(dims_to_reduce), keepdim=True)
            amax = amax.clamp(min=1e-12)

        if scale_storage == "pot":
            amax = 2 ** torch.round(torch.log2(amax))

        x_norm = x_r / amax
        x_q = self.quantize_elemwise(x_norm, round_mode=round_mode,
                                     allow_denorm=allow_denorm)
        x_q = x_q * amax
        return x_q.reshape(x.shape)

    def _quantize_per_bank_sparse(self, x, granularity, round_mode,
                                   allow_denorm=True, scale_storage="pot",
                                   outlier_format=None):
        """BANK dynamic sparse: per-bank outlier/normal split.

        Within each bank, top-k elements by magnitude (outliers) and the
        remaining elements (normals) each get their own per-bank amax.
        Uses the TWO_GROUP format: both groups are quantized with the same
        elemwise quantizer — only the scale differs.
        """
        if self.ebits > 0:
            return self._quantize_per_bank(x, granularity, round_mode,
                                           allow_denorm=allow_denorm,
                                           scale_storage=scale_storage)

        axis = granularity.bank_axis
        if axis < 0:
            axis = x.ndim + axis
        if not (0 <= axis < x.ndim):
            raise ValueError(
                f"bank_axis={granularity.bank_axis} out of range "
                f"for tensor with ndim={x.ndim}"
            )

        bank_size = granularity.bank_size
        N_along = x.shape[axis]
        if N_along % bank_size != 0:
            raise ValueError(
                f"Dimension {axis} size {N_along} not divisible "
                f"by bank_size {bank_size}"
            )

        num_banks = N_along // bank_size
        new_shape = list(x.shape)
        new_shape[axis] = num_banks
        new_shape.insert(axis + 1, bank_size)
        x_r = x.reshape(new_shape)  # (..., num_banks, bank_size, ...)

        # Transpose bank dim to front for per-group top-k
        ndim_r = x_r.ndim
        perm = list(range(ndim_r))
        perm.pop(axis)
        perm = [axis] + perm
        x_b = x_r.permute(perm)  # (num_banks, ..., bank_size, ...)

        group_size = x_b[0].numel()
        k = max(1, int(group_size * granularity.outlier_ratio))
        if k >= group_size:
            # Degenerate: all elements are outliers → standard per_bank
            return self._quantize_per_bank(x, granularity, round_mode,
                                           allow_denorm=allow_denorm,
                                           scale_storage=scale_storage)

        x_flat = x_b.reshape(num_banks, group_size)
        _, top_indices = torch.topk(torch.abs(x_flat), k, dim=1)
        mask_flat = torch.zeros(num_banks, group_size, dtype=torch.bool, device=x.device)
        mask_flat.scatter_(1, top_indices, True)
        mask_b = mask_flat.reshape(x_b.shape)

        # Undo permutation: back to (..., num_banks, bank_size, ...)
        inv_perm = [0] * ndim_r
        for i, p in enumerate(perm):
            inv_perm[p] = i
        mask_r = mask_b.permute(inv_perm)  # back to reshaped layout

        # Per-bank per-group amax — reduce all dims except bank dim (axis)
        dims_to_reduce = [i for i in range(x_r.ndim) if i != axis]

        amax_o = torch.amax(torch.abs(x_r * mask_r.float()), dim=tuple(dims_to_reduce),
                            keepdim=True).clamp(min=1e-12)
        amax_n = torch.amax(torch.abs(x_r * (~mask_r).float()), dim=tuple(dims_to_reduce),
                            keepdim=True).clamp(min=1e-12)

        if scale_storage == "pot":
            amax_o = 2 ** torch.round(torch.log2(amax_o))
            amax_n = 2 ** torch.round(torch.log2(amax_n))

        q_fmt = outlier_format if outlier_format is not None else self
        x_q = torch.zeros_like(x_r)

        # Outlier group
        x_q_o = q_fmt.quantize_elemwise(
            x_r / amax_o, round_mode=round_mode, allow_denorm=allow_denorm)
        x_q = x_q + x_q_o * amax_o * mask_r.float()

        # Normal group
        x_q_n = self.quantize_elemwise(
            x_r / amax_n, round_mode=round_mode, allow_denorm=allow_denorm)
        x_q = x_q + x_q_n * amax_n * (~mask_r).float()

        # Preserve special values
        x_q[x_r == float("Inf")] = float("Inf")
        x_q[x_r == -float("Inf")] = -float("Inf")
        x_q[x_r == float("NaN")] = float("NaN")

        return x_q.reshape(x.shape)

    def _per_block_norm_shift(self) -> int:
        """Extra shared-exponent shift for per-block normalization.

        Per-block normalizes by ``2^shared_exp`` where shared_exp is
        ``floor(log2(amax))``.  After normalization the max magnitude is
        in [1, 2).  Formats whose ``max_norm < 2`` need an extra shift
        so the normalized values fit within the representable range.

        Returns 0 for most formats.  Subclasses (e.g. LUT formats) may
        return a positive integer to shift the normalization factor.
        """
        return 0

    def _quantize_per_block(self, x, granularity, round_mode, scale=None,
                              scale_storage="pot",
                              _shared_exp_method="max",
                              _flush_fp32_subnorms=False,
                              outlier_format=None):
        """Per-block quantization with MX-style shared exponents.

        Same structure as _quantize_per_channel:
        tile into blocks → compute shared exponent → normalize →
        elemwise quantize → rescale → until back to original shape.

        MX shared exponents are inherently power-of-two; ``scale_storage``
        has no effect in this path.

        During ONNX export (both TorchScript and Dynamo-based), return x
        unchanged — the Function's symbolic() method handles quantization
        in the ONNX graph.
        """
        if torch.jit.is_tracing():
            return x
        from src.session._context import _onnx_export_active
        if _onnx_export_active.get():
            return x

        if granularity.outlier_ratio > 0.0:
            from src.formats._outlier_utils import _quantize_outlier_bank
            return _quantize_outlier_bank(
                self, x, granularity, round_mode, scale_storage=scale_storage,
                outlier_format=outlier_format)

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
        shared_exp = shared_exp - self.emax + self._per_block_norm_shift()

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

    def _quantize_static_sparse(self, x, mask, amax_n, amax_o, round_mode,
                                 allow_denorm=True, scale_storage="pot",
                                 outlier_format=None):
        """Static sparse quantization with pre-computed mask and scales.

        Works for all granularity modes — mask and scale shapes are
        broadcasting-compatible with x (determined by caller).

        Args:
            x: Input tensor.
            mask: Boolean mask, True = outlier. Same shape as x.
            amax_n: Normal group amax (shape matches granularity).
            amax_o: Outlier group amax (shape matches granularity).
            round_mode: Quantization rounding mode.
            allow_denorm: Allow denormal values in output.
            scale_storage: "pot" or "fp32" for scale rounding.
            outlier_format: If set, quantize outlier group with this format.

        Returns:
            Quantized tensor with same shape as x.
        """
        amax_o = amax_o.clamp(min=1e-12)
        amax_n = amax_n.clamp(min=1e-12)

        if scale_storage == "pot":
            amax_o = 2 ** torch.round(torch.log2(amax_o))
            amax_n = 2 ** torch.round(torch.log2(amax_n))

        # Outlier group — mask True elements
        x_o = x * mask.float()
        x_o_norm = x_o / amax_o
        q_fmt = outlier_format if outlier_format is not None else self
        x_q_o = q_fmt.quantize_elemwise(x_o_norm, round_mode=round_mode,
                                         allow_denorm=allow_denorm)
        x_q = x_q_o * amax_o * mask.float()

        # Normal group — mask False elements
        x_n = x * (~mask).float()
        x_n_norm = x_n / amax_n
        x_q_n = self.quantize_elemwise(x_n_norm, round_mode=round_mode,
                                        allow_denorm=allow_denorm)
        x_q = x_q + x_q_n * amax_n * (~mask).float()

        # Preserve special values
        x_q[x == float("Inf")] = float("Inf")
        x_q[x == -float("Inf")] = -float("Inf")
        x_q[x == float("NaN")] = float("NaN")

        return x_q

    def _quantize_per_bank_static_sparse(self, x, mask, amax_n, amax_o, granularity,
                                          round_mode, allow_denorm=True, scale_storage="pot",
                                          outlier_format=None):
        """BANK static sparse: reshape to expose bank dim, then common static path.

        Reshapes x and mask to (..., num_banks, bank_size) so that per-bank
        amax tensors broadcast correctly with the reshaped tensor.
        """
        axis = granularity.bank_axis
        if axis < 0:
            axis = x.ndim + axis
        if not (0 <= axis < x.ndim):
            raise ValueError(
                f"bank_axis={granularity.bank_axis} out of range "
                f"for tensor with ndim={x.ndim}"
            )
        bank_size = granularity.bank_size
        N_along = x.shape[axis]
        if N_along % bank_size != 0:
            raise ValueError(
                f"Dimension {axis} size {N_along} not divisible "
                f"by bank_size {bank_size}"
            )

        num_banks = N_along // bank_size

        # Reshape: split axis into (num_banks, bank_size)
        new_shape = list(x.shape)
        new_shape[axis] = num_banks
        new_shape.insert(axis + 1, bank_size)
        x_r = x.reshape(new_shape)
        # Mask may have smaller batch dim than x (calibration batch=1 vs inference
        # batch>1). Expand to match x's shape before reshaping.
        if mask.shape != x.shape:
            mask = mask.expand(x.shape)
        mask_r = mask.reshape(new_shape)

        # Ensure amax tensors are broadcastable with x_r.
        # x_r has bank dim at `axis` and bank_size dim at `axis+1`.
        # amax should have size 1 on all dims except `axis` where it has num_banks.
        # Scalars (ndim==0) broadcast naturally — skip reshape.
        target_shape = [1] * x_r.ndim
        target_shape[axis] = num_banks
        for name, t in [("amax_n", amax_n), ("amax_o", amax_o)]:
            if t.ndim > 0 and t.numel() != num_banks:
                raise ValueError(
                    f"{name} has {t.numel()} elements but {num_banks} banks "
                    f"are expected. Shape {tuple(t.shape)} cannot be reshaped "
                    f"to target {tuple(target_shape)}."
                )
            if t.ndim > 0 and (t.ndim != x_r.ndim or t.shape != torch.Size(target_shape)):
                if name == "amax_n":
                    amax_n = amax_n.reshape(target_shape)
                else:
                    amax_o = amax_o.reshape(target_shape)

        result = self._quantize_static_sparse(
            x_r, mask_r, amax_n, amax_o, round_mode,
            allow_denorm=allow_denorm, scale_storage=scale_storage,
            outlier_format=outlier_format,
        )
        return result.reshape(x.shape)

    def _quantize_per_tensor_sparse(self, x, granularity, round_mode,
                                     allow_denorm=True, scale_storage="pot",
                                     outlier_format=None):
        """Per-tensor quantization with outlier/normal split.

        Splits the tensor into top-k outliers (by magnitude) and normals.
        Each group gets its own per-tensor amax.  Degenerates to standard
        per_tensor when k >= numel.

        If ``outlier_format`` is set, the outlier group is quantized with
        that format instead of ``self``.
        """
        # Float formats: sparse normalization is redundant with the format's
        # native dynamic range; delegate to non-sparse direct elemwise path.
        if self.ebits > 0:
            return self._quantize_per_tensor(x, round_mode, allow_denorm=allow_denorm,
                                              scale_storage=scale_storage)

        N = x.numel()
        k = max(1, int(N * granularity.outlier_ratio))
        if k >= N:
            return self._quantize_per_tensor(x, round_mode, allow_denorm=allow_denorm,
                                              scale_storage=scale_storage)

        # Top-k by magnitude
        _, top_indices = torch.topk(torch.abs(x).flatten(), k)
        mask_flat = torch.zeros(N, dtype=torch.bool, device=x.device)
        mask_flat.scatter_(0, top_indices, True)
        mask = mask_flat.reshape(x.shape)

        # Per-group amax
        amax_o = torch.amax(torch.abs(x * mask.float()))
        amax_n = torch.amax(torch.abs(x * (~mask).float()))

        amax_o = amax_o.clamp(min=1e-12)
        amax_n = amax_n.clamp(min=1e-12)

        if scale_storage == "pot":
            amax_o = 2 ** torch.round(torch.log2(amax_o))
            amax_n = 2 ** torch.round(torch.log2(amax_n))

        # Quantize each group separately
        x_q = torch.zeros_like(x)
        q_fmt = outlier_format if outlier_format is not None else self

        # Outlier group
        x_o = x * mask.float()
        x_o_norm = x_o / amax_o
        x_q_o = q_fmt.quantize_elemwise(
            x_o_norm, round_mode=round_mode, allow_denorm=allow_denorm)
        x_q = x_q + x_q_o * amax_o * mask.float()

        # Normal group
        x_n = x * (~mask).float()
        x_n_norm = x_n / amax_n
        x_q_n = self.quantize_elemwise(
            x_n_norm, round_mode=round_mode, allow_denorm=allow_denorm)
        x_q = x_q + x_q_n * amax_n * (~mask).float()

        # Preserve special values
        x_q[x == float("Inf")] = float("Inf")
        x_q[x == -float("Inf")] = -float("Inf")
        x_q[x == float("NaN")] = float("NaN")

        return x_q

    def _quantize_per_channel_sparse(self, x, granularity, round_mode,
                                      allow_denorm=True, scale_storage="pot",
                                      outlier_format=None):
        """Per-channel quantization with per-channel outlier/normal split.

        Within each channel, the top-k elements by magnitude (outliers) and
        the remaining elements (normals) each get their own per-channel amax.
        Degenerates to standard per_channel when k >= elements_per_channel.

        If ``outlier_format`` is set, the outlier group is quantized with
        that format instead of ``self``.
        """
        axis = granularity.channel_axis
        if axis < 0:
            axis = x.ndim + axis

        # Float formats: sparse normalization is redundant with the format's
        # native dynamic range; delegate to non-sparse path.
        if self.ebits > 0:
            return self._quantize_per_channel(x, granularity, round_mode,
                                               allow_denorm=allow_denorm,
                                               scale_storage=scale_storage)

        C = x.shape[axis]

        # Transpose channel to dim 0 for per-channel iteration
        x_t = x.transpose(0, axis)
        shape_0 = x_t.shape  # (C, ...)
        N_per_channel = x_t[0].numel()
        k = max(1, int(N_per_channel * granularity.outlier_ratio))
        if k >= N_per_channel:
            return self._quantize_per_channel(x, granularity, round_mode,
                                               allow_denorm=allow_denorm,
                                               scale_storage=scale_storage)

        # Flatten non-channel dims
        x_flat = x_t.reshape(C, N_per_channel)

        # Per-channel top-k
        _, top_indices = torch.topk(torch.abs(x_flat), k, dim=1)
        mask_flat = torch.zeros(C, N_per_channel, dtype=torch.bool, device=x.device)
        mask_flat.scatter_(1, top_indices, True)

        # Per-channel per-group amax
        x_masked_o = x_flat * mask_flat.float()
        x_masked_n = x_flat * (~mask_flat).float()

        amax_o = torch.amax(torch.abs(x_masked_o), dim=1)  # (C,)
        amax_n = torch.amax(torch.abs(x_masked_n), dim=1)  # (C,)
        amax_o = amax_o.clamp(min=1e-12)
        amax_n = amax_n.clamp(min=1e-12)

        if scale_storage == "pot":
            amax_o = 2 ** torch.round(torch.log2(amax_o))
            amax_n = 2 ** torch.round(torch.log2(amax_n))

        # Reshape amax for broadcasting: (C,) → (C, 1, 1, ...)
        broadcast_shape = (C,) + (1,) * (x_t.ndim - 1)
        amax_o = amax_o.reshape(broadcast_shape)
        amax_n = amax_n.reshape(broadcast_shape)
        mask = mask_flat.reshape(shape_0)

        # Quantize each group
        x_t_q = torch.zeros_like(x_t)
        q_fmt = outlier_format if outlier_format is not None else self

        x_t_o = x_t * mask.float()
        x_t_n = x_t * (~mask).float()

        x_t_q_o = q_fmt.quantize_elemwise(
            x_t_o / amax_o, round_mode=round_mode, allow_denorm=allow_denorm)
        x_t_q = x_t_q_o * amax_o * mask.float()

        x_t_q_n = self.quantize_elemwise(
            x_t_n / amax_n, round_mode=round_mode, allow_denorm=allow_denorm)
        x_t_q = x_t_q + x_t_q_n * amax_n * (~mask).float()

        # Preserve special values
        x_t_q[x_t == float("Inf")] = float("Inf")
        x_t_q[x_t == -float("Inf")] = -float("Inf")
        x_t_q[x_t == float("NaN")] = float("NaN")

        # Undo transpose
        return x_t_q.transpose(0, axis)

    def export_onnx(self, g, x, scheme):
        """Emit unified three-axis ONNX nodes: Scale → Quantize.

        Scale node represents the granularity axis.
        Quantize node represents the format axis.
        Subclasses with non-standard quantize (Truncate, NF4) are handled
        by ``_emit_format_node`` in helpers — no override needed.
        """
        from src.onnx.helpers import _emit_scale_node, _emit_format_node
        scale = _emit_scale_node(g, x, scheme.granularity)
        return _emit_format_node(g, x, scale, self)

    @staticmethod
    def from_str(s: str) -> "FormatBase":
        """Factory: look up format by string name in the registry."""
        from .registry import get_format
        return get_format(s)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, ebits={self.ebits}, mbits={self.mbits})"
