"""QuantConfig: user-facing quantization configuration for session workflow.

Translates user-friendly configuration strings (e.g. "int8", "per_tensor",
"hadamard") into internal OpQuantConfig via :meth:`QuantConfig.to_op_config`.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.formats.base import FormatBase
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform, TransformBase
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform

_VALID_GRANULARITIES = frozenset({"per_tensor", "per_channel", "per_block", "bank"})
_VALID_TRANSFORMS = frozenset({"none", "hadamard", "smoothquant", "prescale", "adaptive"})
_VALID_CALIBRATORS = frozenset({"mse", "max", "percentile", "kl"})
_VALID_SCALE_STORAGES = frozenset({"fp32", "pot"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_granularity(
    granularity: str,
    block_size: Optional[int] = None,
    axis: int = -1,
    outlier_ratio: float = 0.0,
) -> GranularitySpec:
    """Convert string granularity + optional block_size to GranularitySpec."""
    if granularity == "per_tensor":
        return GranularitySpec(mode=GranularitySpec.per_tensor().mode,
                               outlier_ratio=outlier_ratio)
    elif granularity == "per_channel":
        return GranularitySpec(mode=GranularitySpec.per_channel(axis=axis).mode,
                               channel_axis=axis, outlier_ratio=outlier_ratio)
    elif granularity == "per_block":
        if block_size is None:
            raise ValueError("per_block granularity requires block_size")
        return GranularitySpec(mode=GranularitySpec.per_block(size=block_size, axis=axis).mode,
                               block_size=block_size, block_axis=axis,
                               outlier_ratio=outlier_ratio)
    elif granularity == "bank":
        if block_size is None:
            raise ValueError("bank granularity requires bank_size (pass as block_size)")
        return GranularitySpec(
            mode=GranularityMode.BANK,
            bank_size=block_size,
            bank_axis=axis,
            outlier_ratio=outlier_ratio,
        )
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def _make_weight_transform(transform: str) -> TransformBase:
    """Resolve the weight-side transform from a string name."""
    if transform in ("none", "adaptive"):
        return IdentityTransform()
    elif transform == "hadamard":
        return HadamardTransform()
    elif transform in ("prescale", "smoothquant"):
        return IdentityTransform()
    else:
        raise ValueError(f"Unknown transform: {transform}")


def _make_activation_transform(transform: str, sq_alpha: float) -> TransformBase:
    """Resolve the activation-side transform from a string name.

    SmoothQuantTransform is created with a dummy scale tensor since the
    real per-channel scale comes from calibration later.  This placeholder
    is sufficient for OpQuantConfig construction and scheme resolution.
    """
    if transform in ("none", "adaptive"):
        return IdentityTransform()
    elif transform == "hadamard":
        return HadamardTransform()
    elif transform == "prescale":
        return IdentityTransform()
    elif transform == "smoothquant":
        import torch  # local — avoid module-level torch dependency for one tensor

        # Dummy scale — replaced by calibrated scale during the session workflow.
        return SmoothQuantTransform(torch.tensor([1.0]), channel_axis=-1)
    else:
        raise ValueError(f"Unknown transform: {transform}")


def _make_storage_scheme(width: int, kind: str, scale_storage: str) -> QuantScheme:
    """Build an element-wise storage QuantScheme from bfloat/fp width.

    Args:
        width: Bit width (16 for bfloat16, 8 for fp8, etc.)
        kind: "bfloat" or "fp"
        scale_storage: "fp32" or "pot"

    Returns:
        QuantScheme with per_tensor granularity — the format for
        element-wise (storage) quantization applied to every tensor.
    """
    if kind == "bfloat":
        if width == 16:
            from src.formats.bf16_fp16 import BFloat16Format
            fmt = BFloat16Format()
        else:
            from src.formats.fp_formats import FPFormat
            mbits = width - 7
            fmt = FPFormat(name=f"bfloat{width}", ebits=8, mbits=mbits)
    elif kind == "fp":
        from src.formats.fp_formats import FPFormat
        mantissa_bits = width - 6
        mbits = mantissa_bits + 2
        fmt = FPFormat(name=f"fp{width}", ebits=5, mbits=mbits)
    else:
        raise ValueError(f"Unknown storage kind: {kind}")

    return QuantScheme(
        format=fmt,
        granularity=GranularitySpec.per_tensor(),
        scale_storage=scale_storage,
    )


# ---------------------------------------------------------------------------
# QuantConfig
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """User-facing quantization configuration for the session workflow.

    All fields have defaults, so constructing ``QuantConfig()`` gives a
    sensible baseline (INT8 per-tensor, no transform).  Override only the
    fields you need to change.

    Translates to :class:`OpQuantConfig` via :meth:`to_op_config`, which
    resolves format/granularity/transform strings into concrete objects
    and assembles the per-role ``QuantScheme`` instances.
    """

    # ---- Identity / naming ----
    name: str = ""

    # ---- Weight quantization ----
    w_format: str = "int8"
    w_granularity: str = "per_tensor"
    w_block_size: Optional[int] = None
    w_axis: int = -1

    # ---- Activation quantization (None = same format as weight) ----
    a_format: Optional[str] = None
    a_granularity: str = "per_tensor"
    a_block_size: Optional[int] = None
    a_axis: int = -1

    # ---- Transform ----
    transform: str = "none"               # none | hadamard | smoothquant | prescale
    sq_alpha: float = 0.5
    prescale_init: str = "ones"           # ones | amax | pot_amax
    prescale_pot: bool = False
    prescale_granularity: Optional[str] = None  # None = follow a_granularity

    # ---- LSQ (learned step-size quantization, only valid with prescale) ----
    lsq_steps: int = 0
    lsq_lr: float = 1e-3

    # ---- Scale storage format (QuantScheme.scale_storage) ----
    scale_storage: str = "pot"           # pot | fp32

    # ---- Calibrator ----
    calibrator: str = "mse"              # mse | max | percentile | kl

    # ---- Element-wise storage quantization ----
    storage_bits: int = 0                   # 16 = bfloat16, 8 = fp8, etc. 0 = disabled
    storage_kind: str = "bfloat"            # "bfloat" | "fp"
    storage_format: Optional[str] = None    # Explicit format name: "fp8_e4m3", "fp4_e2m1", etc.
                                            # Takes precedence over storage_bits/storage_kind.

    # ---- GPTQ (Hessian-based weight-only quantization) ----
    gptq: bool = False
    gptq_block_size: int = 128
    gptq_damp: float = 0.01
    gptq_act_order: bool = False

    # ---- Sparse (outlier bank) ----
    outlier_ratio: float = 0.0              # ∈ [0, 1). >0: split each granularity group into outliers + normals
    outlier_format: Optional[str] = None    # Format for outlier group (None = use main format)
    a_outlier_format: Optional[str] = None  # Activation-only override (None = follow outlier_format)

    # ---- Mode ----
    weight_only: bool = False
    quantize_nonlinear: bool = True         # False = skip quantizing nonlinear ops (norm/activation/pool)
    static_input_scale: bool = False        # True → use calibrated _input_scale for input activation quant

    def __post_init__(self):
        # prescale_granularity only matters when transform='prescale'
        if self.transform == "prescale" and self.prescale_granularity is None:
            # Pre-scale cannot operate at per_block: it is a per-channel or
            # per-tensor scaling factor.  Map per_block → per_channel.
            if self.a_granularity == "per_block":
                self.prescale_granularity = "per_channel"
            else:
                self.prescale_granularity = self.a_granularity

        if self.w_granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"Invalid w_granularity {self.w_granularity!r}. "
                f"Must be one of {sorted(_VALID_GRANULARITIES)}"
            )
        if self.a_granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"Invalid a_granularity {self.a_granularity!r}. "
                f"Must be one of {sorted(_VALID_GRANULARITIES)}"
            )
        if self.transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"Invalid transform {self.transform!r}. "
                f"Must be one of {sorted(_VALID_TRANSFORMS)}"
            )
        if self.calibrator not in _VALID_CALIBRATORS:
            raise ValueError(
                f"Invalid calibrator {self.calibrator!r}. "
                f"Must be one of {sorted(_VALID_CALIBRATORS)}"
            )
        if self.scale_storage not in _VALID_SCALE_STORAGES:
            raise ValueError(
                f"Invalid scale_storage {self.scale_storage!r}. "
                f"Must be one of {sorted(_VALID_SCALE_STORAGES)}"
            )
        if self.transform == "prescale" and self.prescale_granularity not in ("per_tensor", "per_channel"):
            raise ValueError(
                f"prescale_granularity must be 'per_tensor' or 'per_channel', "
                f"got {self.prescale_granularity!r}. per_block is not supported "
                f"for pre-scale (use 'per_channel' instead)."
            )
        if self.gptq:
            if self.gptq_block_size < 1:
                raise ValueError(
                    f"gptq_block_size must be >= 1, got {self.gptq_block_size}"
                )
            if self.gptq_damp <= 0 or self.gptq_damp > 1:
                raise ValueError(
                    f"gptq_damp must be in (0, 1], got {self.gptq_damp}"
                )
        if self.lsq_steps < 0:
            raise ValueError(
                f"lsq_steps must be >= 0, got {self.lsq_steps}"
            )
        if self.weight_only and self.a_format is not None:
            raise ValueError(
                "a_format cannot be set when weight_only=True"
            )
        if self.transform != "prescale" and self.lsq_steps > 0:
            raise ValueError(
                f"lsq_steps > 0 requires transform='prescale', "
                f"got transform={self.transform!r}"
            )
        if self.w_granularity == "per_block" and self.w_block_size is None:
            raise ValueError(
                "w_block_size is required when w_granularity='per_block'"
            )
        if self.a_granularity == "per_block" and self.a_block_size is None:
            raise ValueError(
                "a_block_size is required when a_granularity='per_block'"
            )
        if self.w_granularity == "bank" and self.w_block_size is None:
            raise ValueError(
                "w_block_size is required when w_granularity='bank' (used as bank_size)"
            )
        if self.a_granularity == "bank" and self.a_block_size is None:
            raise ValueError(
                "a_block_size is required when a_granularity='bank' (used as bank_size)"
            )
        if self.storage_bits < 0:
            raise ValueError(
                f"storage_bits must be >= 0, got {self.storage_bits}"
            )
        if self.storage_kind not in ("bfloat", "fp"):
            raise ValueError(
                f"storage_kind must be 'bfloat' or 'fp', got {self.storage_kind!r}"
            )
        if self.storage_format is not None:
            if not isinstance(self.storage_format, str):
                raise TypeError(
                    f"storage_format must be a string, got {type(self.storage_format).__name__}"
                )
            # Validate that the format name is resolvable
            try:
                FormatBase.from_str(self.storage_format)
            except ValueError as e:
                raise ValueError(
                    f"Unknown storage_format {self.storage_format!r}: {e}"
                ) from None
            if self.storage_bits > 0:
                raise ValueError(
                    "storage_bits cannot be set together with storage_format. "
                    "Use storage_format alone for explicit format names."
                )
        if self.outlier_ratio < 0.0 or self.outlier_ratio > 1.0:
            raise ValueError(
                f"outlier_ratio must be in [0, 1], got {self.outlier_ratio}"
            )
        if self.outlier_format is not None:
            if not isinstance(self.outlier_format, str):
                raise TypeError(
                    f"outlier_format must be a string, got {type(self.outlier_format).__name__}"
                )
            try:
                FormatBase.from_str(self.outlier_format)
            except ValueError as e:
                raise ValueError(
                    f"Unknown outlier_format {self.outlier_format!r}: {e}"
                ) from None
        if self.a_outlier_format is not None:
            if not isinstance(self.a_outlier_format, str):
                raise TypeError(
                    f"a_outlier_format must be a string, got {type(self.a_outlier_format).__name__}"
                )
            try:
                FormatBase.from_str(self.a_outlier_format)
            except ValueError as e:
                raise ValueError(
                    f"Unknown a_outlier_format {self.a_outlier_format!r}: {e}"
                ) from None
            if self.weight_only:
                raise ValueError(
                    "a_outlier_format cannot be set when weight_only=True"
                )

    def to_op_config(self) -> OpQuantConfig:
        """Convert this user-facing config to internal :class:`OpQuantConfig`.

        Resolution order:
        1. Resolve ``w_format`` / ``a_format`` → :class:`FormatBase` via
           :meth:`FormatBase.from_str`.
        2. Build :class:`GranularitySpec` from the granularity strings.
        3. Select per-role :class:`TransformBase` according to the transform
           rule table (weight and activation differ for smoothquant).
        4. Construct per-role :class:`QuantScheme` and assemble the result.
        """
        # ---- Format ----
        w_fmt = FormatBase.from_str(self.w_format)
        a_fmt = FormatBase.from_str(self.a_format) if self.a_format is not None else w_fmt

        # ---- Granularity ----
        w_gran = _resolve_granularity(self.w_granularity, self.w_block_size, axis=self.w_axis,
                                       outlier_ratio=self.outlier_ratio)
        a_gran = _resolve_granularity(self.a_granularity, self.a_block_size, axis=self.a_axis,
                                       outlier_ratio=self.outlier_ratio)

        # ---- Transform (per-role, see rule table in `_make_*` helpers) ----
        w_tx = _make_weight_transform(self.transform)
        a_tx = _make_activation_transform(self.transform, self.sq_alpha)

        # ---- QuantScheme ----
        w_outlier_fmt = FormatBase.from_str(self.outlier_format) if self.outlier_format is not None else None
        a_outlier_fmt_raw = self.a_outlier_format if self.a_outlier_format is not None else self.outlier_format
        a_outlier_fmt = FormatBase.from_str(a_outlier_fmt_raw) if a_outlier_fmt_raw is not None else None

        w_scheme = QuantScheme(
            format=w_fmt,
            granularity=w_gran,
            transform=w_tx,
            scale_storage=self.scale_storage,
            outlier_format=w_outlier_fmt,
        )
        a_scheme = QuantScheme(
            format=a_fmt,
            granularity=a_gran,
            transform=a_tx,
            scale_storage=self.scale_storage,
            outlier_format=a_outlier_fmt,
        )

        # ---- Storage scheme (element-wise) ----
        storage = None
        if self.storage_format is not None:
            storage = QuantScheme(
                format=FormatBase.from_str(self.storage_format),
                granularity=GranularitySpec.per_tensor(),
                scale_storage=self.scale_storage,
            )
        elif self.storage_bits > 0:
            storage = _make_storage_scheme(self.storage_bits, self.storage_kind, self.scale_storage)

        # ---- Assemble OpQuantConfig ----
        if self.weight_only:
            return OpQuantConfig(weight=w_scheme, storage=storage)

        return OpQuantConfig(input=a_scheme, weight=w_scheme, storage=storage)

    @classmethod
    def from_descriptor(cls, desc: Dict[str, Any]) -> "QuantConfig":
        """Create a QuantConfig from a legacy dict descriptor.

        Supports all keys from the old ``resolve_config`` API in
        ``src/pipeline/config.py``:

        * ``format`` (str) → ``w_format``
        * ``act_format`` (str, optional) → ``a_format``
        * ``granularity`` (str) → ``w_granularity``, ``a_granularity``
        * ``block_size`` (int, optional) → ``w_block_size``, ``a_block_size``
        * ``axis`` (int, optional) ― applied to both granularities
        * ``transform`` (str, optional) → ``transform``
        * ``scale_format`` (str) → ``scale_storage`` (maps ``"fp32"/"pot"``)
        * ``weight_only`` (bool) → ``weight_only``
        * ``name`` (str) → ``name``
        * ``lsq_steps`` (int) → ``lsq_steps``
        * ``lsq_lr`` (float) → ``lsq_lr``
        * ``pre_scale_init`` (str) → ``prescale_init``
        * ``pre_scale_pot`` (bool) → ``prescale_pot``
        * ``storage_format`` (str, optional) → ``storage_format`` — explicit
          format name, e.g. ``"fp8_e4m3"`` or ``"fp4_e2m1"``.

        Raises:
            ValueError: When required keys are missing or values are invalid.
            TypeError: When values have the wrong type.
        """
        fmt_name = desc.get("format")
        if fmt_name is not None and not isinstance(fmt_name, str):
            raise TypeError(
                f"'format' must be a string, got {type(fmt_name).__name__}"
            )

        gran = desc.get("granularity")
        if gran is not None and not isinstance(gran, str):
            raise TypeError(
                f"'granularity' must be a string, got {type(gran).__name__}"
            )

        axis = desc.get("axis", -1)
        if not isinstance(axis, int):
            raise TypeError(f"'axis' must be an int, got {type(axis).__name__}")

        block_size = desc.get("block_size")
        if block_size is not None and not isinstance(block_size, int):
            raise TypeError(
                f"'block_size' must be an int, got {type(block_size).__name__}"
            )

        transform = desc.get("transform", "none")
        if transform is not None and not isinstance(transform, str):
            raise TypeError(
                f"'transform' must be a string, got {type(transform).__name__}"
            )

        scale_storage = desc.get("scale_format", "fp32")

        weight_only = desc.get("weight_only", False)
        if not isinstance(weight_only, bool):
            raise TypeError(
                f"'weight_only' must be a bool, got {type(weight_only).__name__}"
            )

        act_format = desc.get("act_format")
        if act_format is not None and not isinstance(act_format, str):
            raise TypeError(
                f"'act_format' must be a string, got {type(act_format).__name__}"
            )
        if weight_only and act_format is not None:
            raise ValueError(
                "'act_format' cannot be used with 'weight_only=True'"
            )

        gptq = desc.get("gptq", False)
        gptq_block_size = desc.get("gptq_block_size", 128)
        gptq_damp = desc.get("gptq_damp", 0.01)
        gptq_act_order = desc.get("gptq_act_order", False)

        lsq_steps = desc.get("lsq_steps", 0)
        lsq_lr = desc.get("lsq_lr", 1e-3)
        prescale_init = desc.get("pre_scale_init", "ones")
        prescale_pot = desc.get("pre_scale_pot", False)

        outlier_ratio = desc.get("outlier_ratio", 0.0)
        outlier_format = desc.get("outlier_format")
        if outlier_format is not None and not isinstance(outlier_format, str):
            raise TypeError(
                f"'outlier_format' must be a string, got {type(outlier_format).__name__}"
            )
        a_outlier_format = desc.get("a_outlier_format")
        if a_outlier_format is not None and not isinstance(a_outlier_format, str):
            raise TypeError(
                f"'a_outlier_format' must be a string, got {type(a_outlier_format).__name__}"
            )
        if weight_only and a_outlier_format is not None:
            raise ValueError(
                "'a_outlier_format' cannot be used with 'weight_only=True'"
            )

        # Storage: support legacy keys "bfloat"/"fp" for backward compat
        _sbits = desc.get("storage_bits", 0)
        _skind = desc.get("storage_kind", "bfloat")
        _sfmt = desc.get("storage_format")
        if _sfmt is not None and not isinstance(_sfmt, str):
            raise TypeError(
                f"'storage_format' must be a string, got {type(_sfmt).__name__}"
            )
        if "bfloat" in desc:
            _sbits = desc["bfloat"]
            _skind = "bfloat"
        elif "fp" in desc:
            _sbits = desc["fp"]
            _skind = "fp"

        # Resolve w_granularity eagerly so axis is threaded through
        w_gran = _resolve_granularity(
            gran or "per_tensor",
            block_size=block_size if isinstance(block_size, int) else None,
            axis=axis,
            outlier_ratio=outlier_ratio,
        )
        a_gran = _resolve_granularity(
            gran or "per_tensor",
            block_size=block_size if isinstance(block_size, int) else None,
            axis=axis,
            outlier_ratio=outlier_ratio,
        )

        return cls(
            name=desc.get("name", ""),
            w_format=fmt_name or "int8",
            a_format=act_format,
            w_granularity=gran or "per_tensor",
            a_granularity=gran or "per_tensor",
            w_block_size=(w_gran.bank_size
                          if w_gran.mode.name == "BANK" else
                          w_gran.block_size
                          if w_gran.mode.name == "PER_BLOCK" else None),
            a_block_size=(a_gran.bank_size
                          if a_gran.mode.name == "BANK" else
                          a_gran.block_size
                          if a_gran.mode.name == "PER_BLOCK" else None),
            w_axis=desc.get("w_axis", axis),
            a_axis=desc.get("a_axis", axis),
            transform=transform,
            scale_storage=scale_storage,
            storage_bits=_sbits,
            storage_kind=_skind,
            storage_format=_sfmt,
            weight_only=weight_only,
            gptq=gptq,
            gptq_block_size=gptq_block_size,
            gptq_damp=gptq_damp,
            gptq_act_order=gptq_act_order,
            lsq_steps=lsq_steps,
            lsq_lr=lsq_lr,
            prescale_init=prescale_init,
            prescale_pot=prescale_pot,
            outlier_ratio=outlier_ratio,
            outlier_format=outlier_format,
            a_outlier_format=a_outlier_format,
        )


def resolve_config(desc: Dict[str, Any]) -> OpQuantConfig:
    """Backward-compat: convert a legacy study descriptor dict to OpQuantConfig.

    Thin wrapper that validates the descriptor's required keys and legacy
    field names, then delegates to :meth:`QuantConfig.from_descriptor` +
    :meth:`QuantConfig.to_op_config`.

    Use ``QuantConfig.from_descriptor(desc).to_op_config()`` directly for
    new code — *resolve_config* exists only for backward compatibility.
    """
    # Required keys (QuantConfig.from_descriptor has defaults, but legacy
    # callers expect strict validation for these two fields).
    if "format" not in desc:
        raise ValueError("descriptor must contain 'format' key")
    if "granularity" not in desc:
        raise ValueError("descriptor must contain 'granularity' key")

    # Legacy field name validation (scale_format → scale_storage)
    scale_storage = desc.get("scale_format", "fp32")
    if not isinstance(scale_storage, str):
        raise TypeError(
            f"'scale_format' must be a string, got {type(scale_storage).__name__}"
        )
    if scale_storage not in ("fp32", "pot"):
        raise ValueError(
            f"Invalid scale_format {scale_storage!r}. Must be 'fp32' or 'pot'"
        )

    return QuantConfig.from_descriptor(desc).to_op_config()
