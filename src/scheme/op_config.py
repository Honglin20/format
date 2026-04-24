"""
OpQuantConfig: operator-level quantization configuration.

Each field is a tuple[QuantScheme, ...] pipeline applied to a specific tensor
role. Empty tuple = no quantization (identity pass-through).
"""
from dataclasses import dataclass, fields
from typing import Tuple

from .quant_scheme import QuantScheme

_BACKWARD_FIELD_NAMES = frozenset((
    "grad_output", "grad_input", "grad_weight", "grad_bias",
    "input_gw", "grad_output_gw", "weight_gi", "grad_output_gi",
))


@dataclass(frozen=True)
class OpQuantConfig:
    """Operator-level quantization configuration.

    Each field is a tuple[QuantScheme, ...] pipeline:
    - Empty tuple = no quantization for that role
    - Single scheme = one quantization step
    - Multiple schemes = chained pipeline: x -> quantize(x, s1) -> quantize(., s2) -> ...

    Default construction (no arguments) produces a configuration with no
    quantization on any role — all 12 pipelines are empty.

    Forward roles: input, weight, bias, output
    Backward roles (QAT): grad_output, grad_input, grad_weight, grad_bias
    Backward gemm re-quantization: input_gw, grad_output_gw, weight_gi, grad_output_gi

    Non-matmul operators (activation, softmax, norm, elemwise) use only
    input/output (+ optional grad_output/grad_input); other fields remain ().
    """

    # ---------- forward ----------
    input:  Tuple[QuantScheme, ...] = ()   # Default: no quantization
    weight: Tuple[QuantScheme, ...] = ()   # Default: no quantization
    bias:   Tuple[QuantScheme, ...] = ()   # Default: no quantization
    output: Tuple[QuantScheme, ...] = ()   # Default: no quantization

    # ---------- backward (QAT) ----------
    grad_output: Tuple[QuantScheme, ...] = ()   # Default: no quantization
    grad_input:  Tuple[QuantScheme, ...] = ()   # Default: no quantization
    grad_weight: Tuple[QuantScheme, ...] = ()   # Default: no quantization
    grad_bias:   Tuple[QuantScheme, ...] = ()   # Default: no quantization

    # ---------- backward gemm re-quantization ----------
    input_gw:       Tuple[QuantScheme, ...] = ()   # Default: no quantization
    grad_output_gw: Tuple[QuantScheme, ...] = ()   # Default: no quantization
    weight_gi:       Tuple[QuantScheme, ...] = ()   # Default: no quantization
    grad_output_gi:  Tuple[QuantScheme, ...] = ()   # Default: no quantization

    def __post_init__(self):
        # All current fields are tuple[QuantScheme, ...] pipelines.
        # If a non-pipeline field is added, this loop needs a type dispatch.
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, tuple):
                raise TypeError(
                    f"OpQuantConfig.{f.name} must be tuple[QuantScheme, ...], "
                    f"got {type(value).__name__}"
                )
            for i, s in enumerate(value):
                if not isinstance(s, QuantScheme):
                    raise TypeError(
                        f"OpQuantConfig.{f.name}[{i}] must be QuantScheme, "
                        f"got {type(s).__name__}"
                    )

    @property
    def is_training(self) -> bool:
        """True if any backward field is non-empty (QAT active)."""
        return any(getattr(self, name) for name in _BACKWARD_FIELD_NAMES)
