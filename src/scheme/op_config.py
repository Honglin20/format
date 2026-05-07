"""
OpQuantConfig: operator-level quantization configuration — two-level model.

Quantization has exactly two types:
- storage: storage precision (per-tensor elemwise cast), uniform across all tensors
- compute: compute quantization (per-block MX etc.), per-role

Each field is QuantScheme | None. No tuples, no pipelines, no iteration.
"""
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from ..formats.base import FormatBase
from .granularity import GranularitySpec
from .quant_scheme import QuantScheme
from .transform import IdentityTransform, TransformBase
from ..transform.hadamard import HadamardTransform

_BACKWARD_FIELD_NAMES = frozenset((
    "grad_output", "grad_input", "grad_weight", "grad_bias",
    "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi",
))


@dataclass(frozen=True)
class OpQuantConfig:
    """Operator-level quantization configuration.

    Two-level quantization model:
    - storage: applied to EVERY tensor at every quantization point,
      always first (elemwise storage precision cast, e.g. bfloat16)
    - compute: role-specific compute quantization (e.g. fp8 MX per-block)

    Default construction (no arguments) = no quantization on any role.
    """

    # ---- Storage (uniform across all tensors in the model) ----
    storage: Optional[QuantScheme] = None

    # ---- Compute quantization (one per role, None = no compute quant) ----
    input:  Optional[QuantScheme] = None
    weight: Optional[QuantScheme] = None
    bias:   Optional[QuantScheme] = None
    output: Optional[QuantScheme] = None

    # ---- Backward (QAT) ----
    grad_output: Optional[QuantScheme] = None
    grad_input:  Optional[QuantScheme] = None
    grad_weight: Optional[QuantScheme] = None
    grad_bias:   Optional[QuantScheme] = None

    # ---- Backward gemm re-quantization ----
    input_gw:       Optional[QuantScheme] = None
    grad_output_gw: Optional[QuantScheme] = None
    weight_gi:       Optional[QuantScheme] = None
    grad_output_gi:  Optional[QuantScheme] = None

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None and not isinstance(value, QuantScheme):
                raise TypeError(
                    f"OpQuantConfig.{f.name} must be QuantScheme or None, "
                    f"got {type(value).__name__}"
                )

    @property
    def is_training(self) -> bool:
        """True if any backward field is non-None (QAT active)."""
        return any(getattr(self, name) is not None for name in _BACKWARD_FIELD_NAMES)

    @classmethod
    def from_descriptor(cls, desc: Dict[str, Any]) -> "OpQuantConfig":
        """Convert a search-space descriptor dict to OpQuantConfig.

        This absorbs the logic previously in ``pipeline/config.py:resolve_config()``.

        The descriptor dict supports these keys:

        ==================  =========  =============================================
        key                 required   description
        ==================  =========  =============================================
        ``format``          yes        Format name (e.g. ``"int8"``, ``"fp8_e4m3"``)
        ``granularity``     yes        ``"per_tensor"`` | ``"per_channel"`` |
                                      ``"per_block"``
        ``axis``            no         Channel/block axis (default ``-1``)
        ``block_size``      conditional  Required for ``per_block``
        ``transform``       no         ``"hadamard"`` | ``"none"`` (default ``None``)
        ``scale_format``    no         ``"fp32"`` | ``"pot"`` (default ``"fp32"``)
        ``act_format``      no         Per-role activation format for mixed precision
        ``weight_only``     no         If ``True``, only weight is quantized
        ==================  =========  =============================================

        Returns:
            ``OpQuantConfig`` with ``input``, ``weight``, ``output`` set
            (or just ``weight`` if ``weight_only``).
        """
        # ---- Format ----
        fmt_name = desc.get("format")
        if fmt_name is None:
            raise ValueError("descriptor must contain 'format' key")
        if not isinstance(fmt_name, str):
            raise TypeError(
                f"'format' must be a string, got {type(fmt_name).__name__}"
            )
        fmt = FormatBase.from_str(fmt_name)

        # ---- Granularity ----
        mode = desc.get("granularity")
        if mode is None:
            raise ValueError("descriptor must contain 'granularity' key")
        if not isinstance(mode, str):
            raise TypeError(
                f"'granularity' must be a string, got {type(mode).__name__}"
            )
        if mode == "per_tensor":
            granularity = GranularitySpec.per_tensor()
        elif mode == "per_channel":
            axis = desc.get("axis", -1)
            if not isinstance(axis, int):
                raise TypeError(f"'axis' must be an int, got {type(axis).__name__}")
            granularity = GranularitySpec.per_channel(axis=axis)
        elif mode == "per_block":
            block_size = desc.get("block_size")
            if block_size is None:
                raise ValueError("per_block granularity requires 'block_size' in descriptor")
            if not isinstance(block_size, int):
                raise TypeError(
                    f"'block_size' must be an int, got {type(block_size).__name__}"
                )
            axis = desc.get("axis", -1)
            if not isinstance(axis, int):
                raise TypeError(f"'axis' must be an int, got {type(axis).__name__}")
            granularity = GranularitySpec.per_block(size=block_size, axis=axis)
        else:
            raise ValueError(f"Unknown granularity: {mode}")

        # ---- Transform ----
        tx = desc.get("transform")
        if tx is None or (isinstance(tx, str) and tx.lower() == "none"):
            transform = IdentityTransform()
        elif isinstance(tx, TransformBase):
            transform = tx
        elif isinstance(tx, str) and tx == "hadamard":
            transform = HadamardTransform()
        elif isinstance(tx, str):
            raise ValueError(f"Unknown transform: {tx}")
        else:
            raise TypeError(
                f"'transform' must be a string, TransformBase, or None, "
                f"got {type(tx).__name__}"
            )

        # ---- Scale format ----
        scale_format = desc.get("scale_format", "fp32")
        if not isinstance(scale_format, str):
            raise TypeError(
                f"'scale_format' must be a string, got {type(scale_format).__name__}"
            )
        if scale_format not in ("fp32", "pot"):
            raise ValueError(
                f"Invalid scale_format {scale_format!r}. Must be 'fp32' or 'pot'"
            )

        # ---- Weight scheme ----
        weight_scheme = QuantScheme(
            format=fmt,
            granularity=granularity,
            transform=transform,
            scale_format=scale_format,
        )

        # ---- Activation scheme (per-role format override) ----
        act_format = desc.get("act_format")
        if act_format is not None:
            if not isinstance(act_format, str):
                raise TypeError(
                    f"'act_format' must be a string, got {type(act_format).__name__}"
                )
            act_fmt = FormatBase.from_str(act_format)
            act_scheme = QuantScheme(
                format=act_fmt,
                granularity=granularity,
                transform=transform,
                scale_format=scale_format,
            )
        else:
            act_scheme = weight_scheme

        # ---- Weight only ----
        weight_only = desc.get("weight_only", False)
        if not isinstance(weight_only, bool):
            raise TypeError(
                f"'weight_only' must be a bool, got {type(weight_only).__name__}"
            )
        if weight_only:
            if act_format is not None:
                raise ValueError(
                    "'act_format' cannot be used with 'weight_only=True'"
                )
            return cls(weight=weight_scheme)

        return cls(input=act_scheme, weight=weight_scheme, output=act_scheme)
