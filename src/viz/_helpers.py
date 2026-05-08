"""Internal helpers shared between viz sub-modules."""
from typing import Dict


def _compute_best_transform_per_layer(
    variant_qsnr: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """Return ``{layer_name: best_transform_name}`` by QSNR.

    For each layer, picks the transform variant (one of the dict keys in
    ``variant_qsnr``) that maximizes per-layer QSNR.  Ties go to the
    first transform encountered in dict insertion order.
    """
    all_layers: set = set()
    for qsnr_dict in variant_qsnr.values():
        all_layers.update(qsnr_dict.keys())
    result: Dict[str, str] = {}
    tx_names = list(variant_qsnr.keys())
    for layer in all_layers:
        result[layer] = max(
            tx_names,
            key=lambda tx, l=layer: variant_qsnr[tx].get(l, -float("inf")),
        )
    return result
