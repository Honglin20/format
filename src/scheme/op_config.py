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

    Forward roles: input, weight, bias, output
    Backward roles (QAT): grad_output, grad_input, grad_weight, grad_bias
    Backward gemm re-quantization: input_gw, grad_output_gw, weight_gi, grad_output_gi

    Non-matmul operators (activation, softmax, norm, elemwise) use only
    input/output (+ optional grad_output/grad_input); other fields remain ().
    """

    # ---------- forward ----------
    input:  Tuple[QuantScheme, ...] = ()
    weight: Tuple[QuantScheme, ...] = ()
    bias:   Tuple[QuantScheme, ...] = ()
    output: Tuple[QuantScheme, ...] = ()

    # ---------- backward (QAT) ----------
    grad_output: Tuple[QuantScheme, ...] = ()
    grad_input:  Tuple[QuantScheme, ...] = ()
    grad_weight: Tuple[QuantScheme, ...] = ()
    grad_bias:   Tuple[QuantScheme, ...] = ()

    # ---------- backward gemm re-quantization ----------
    input_gw:       Tuple[QuantScheme, ...] = ()
    grad_output_gw: Tuple[QuantScheme, ...] = ()
    weight_gi:       Tuple[QuantScheme, ...] = ()
    grad_output_gi:  Tuple[QuantScheme, ...] = ()

    def __post_init__(self):
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
