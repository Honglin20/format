"""Output specifications — declare what output to produce, derive what's needed.

This module uses ONLY string keys for observer names. Observer class resolution
happens in session/_session.py, NOT here.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Output specifications
# ---------------------------------------------------------------------------
# Each entry declares:
#   observers:   list of observer key strings needed by this output
#   needs_eval:  whether this output requires a full evaluation pass
#   needs_cost:  whether this output requires cost model execution
# ---------------------------------------------------------------------------

_OUTPUT_SPEC: Dict[str, dict] = {
    # ── Tables ──────────────────────────────────────────────────────────
    "accuracy":         {"observers": [],              "needs_eval": True},
    "sensitivity":      {"observers": ["qsnr"],        "needs_eval": True},
    "pot_delta":        {"observers": [],              "needs_eval": True},
    "transform_matrix": {"observers": ["qsnr"],        "needs_eval": True},
    "transform_dist":   {"observers": ["qsnr"],        "needs_eval": True},
    # ── Charts ──────────────────────────────────────────────────────────
    "qsnr":             {"observers": ["qsnr"],        "needs_eval": False},
    "mse":              {"observers": ["mse"],         "needs_eval": False},
    "histogram":        {"observers": ["histogram"],   "needs_eval": False},
    "error_dist":       {"observers": ["distribution", "mse"], "needs_eval": False},
    "transform_heatmap": {"observers": ["qsnr"],       "needs_eval": True},
    "transform_pie":     {"observers": ["qsnr"],       "needs_eval": True},
    "transform_delta":   {"observers": ["qsnr", "mse"], "needs_eval": True},
    "layer_qsnr":        {"observers": ["qsnr"],       "needs_eval": False},
    "block_sweep":       {"observers": ["qsnr"],       "needs_eval": True},
    "hierarchical":      {"observers": ["qsnr", "mse"], "needs_eval": True},
    "pot_delta_bar":     {"observers": [],             "needs_eval": True},
    # ── Other ───────────────────────────────────────────────────────────
    "cost":             {"observers": [],              "needs_eval": False, "needs_cost": True},
}

PRESETS: Dict[str, List[str]] = {
    "default": ["accuracy", "qsnr"],
    "all": list(_OUTPUT_SPEC.keys()),
}


def resolve_outputs(
    output_keys: Union[str, List[str]],
) -> Tuple[Set[str], bool, bool]:
    """Resolve output key strings into what the session workflow needs.

    Args:
        output_keys: One of ``"default"``, ``"all"``, or a list of output
            key strings (e.g. ``["accuracy", "qsnr"]``).

    Returns:
        ``(observer_keys: set[str], needs_eval: bool, needs_cost: bool)``

    Raises:
        ValueError: If any output key is unknown.
    """
    if output_keys == "default":
        keys = PRESETS["default"]
    elif output_keys == "all":
        keys = PRESETS["all"]
    else:
        keys = list(output_keys)

    # Validate keys
    valid_keys = set(_OUTPUT_SPEC.keys())
    unknown = [k for k in keys if k not in valid_keys]
    if unknown:
        raise ValueError(
            f"Unknown output key(s): {unknown}. "
            f"Valid keys: {sorted(valid_keys)}"
        )

    obs: Set[str] = set()
    needs_eval = False
    needs_cost = False
    for key in keys:
        spec = _OUTPUT_SPEC[key]
        obs.update(spec["observers"])
        needs_eval = needs_eval or spec.get("needs_eval", False)
        needs_cost = needs_cost or spec.get("needs_cost", False)
    return obs, needs_eval, needs_cost
