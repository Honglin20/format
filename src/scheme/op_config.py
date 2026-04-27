"""
OpQuantConfig: operator-level quantization configuration — two-level model.

Quantization has exactly two types:
- storage: storage precision (per-tensor elemwise cast), uniform across all tensors
- compute: compute quantization (per-block MX etc.), per-role

Each field is QuantScheme | None. No tuples, no pipelines, no iteration.
"""
from dataclasses import dataclass, fields
from typing import Optional

from .quant_scheme import QuantScheme

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
