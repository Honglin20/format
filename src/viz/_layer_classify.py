"""Layer-type classification for operator-type filtering in visualisations."""

from __future__ import annotations

from typing import List, Optional

# ── Pattern tables ──────────────────────────────────────────────────────────

_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("linear",     ["linear", "fc", "dense", "proj", "mlp"]),
    ("conv",       ["conv"]),
    ("norm",       ["norm", "bn", "layernorm", "batchnorm", "instancenorm", "rms"]),
    ("activation", ["relu", "gelu", "silu", "swish", "mish", "tanh", "sigmoid"]),
    ("pool",       ["pool", "avgpool", "maxpool"]),
    ("embedding",  ["embed", "token"]),
]

_VALID_TYPES = frozenset(t for t, _ in _TYPE_PATTERNS) | {"other"}


# ── Public API ──────────────────────────────────────────────────────────────

def classify_layer_type(name: str) -> str:
    """Return the coarse operator type for a module name.

    Args:
        name: Module name string (e.g. ``"fc1"``, ``"conv2"``, ``"bn1"``).

    Returns:
        One of ``"linear"``, ``"conv"``, ``"norm"``, ``"activation"``,
        ``"pool"``, ``"embedding"``, or ``"other"``.
    """
    lower = name.lower()
    for op_type, patterns in _TYPE_PATTERNS:
        if any(p in lower for p in patterns):
            return op_type
    return "other"


def filter_layers_by_type(
    layers: List[str],
    op_types: Optional[List[str]] = None,
) -> List[str]:
    """Filter a list of layer names by operator type.

    Args:
        layers: Layer names to filter.
        op_types: Operator types to keep. ``None`` or empty → keep all.
            Valid values: ``"linear"``, ``"conv"``, ``"norm"``,
            ``"activation"``, ``"pool"``, ``"embedding"``, ``"other"``.

    Returns:
        Filtered list preserving original order.
    """
    if not op_types:
        return list(layers)
    invalid = set(op_types) - _VALID_TYPES
    if invalid:
        raise ValueError(
            f"Invalid operator types: {sorted(invalid)}. "
            f"Valid: {sorted(_VALID_TYPES)}."
        )
    op_set = frozenset(op_types)
    return [n for n in layers if classify_layer_type(n) in op_set]
