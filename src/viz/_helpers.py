"""Internal helpers shared between viz sub-modules."""
from typing import Dict


def _compute_best_transform_per_layer(
    variant_qsnr: Dict[str, Dict[str, float]],
    *,
    eps: float = 0.1,
    prefer: tuple = ("none", "hadamard", "smoothquant"),
) -> Dict[str, str]:
    """Return ``{layer_name: best_transform_name}`` by QSNR.

    For each layer, picks the transform variant that maximizes per-layer
    QSNR. When multiple transforms are within *eps* dB of the best, the
    one appearing earlier in *prefer* (case-insensitive substring match)
    wins — simpler transforms are preferred when noise-equivalent.

    Args:
        variant_qsnr: ``{transform_name: {layer: qsnr_db}}``.
        eps: Tie-breaking threshold in dB (default 0.1).
        prefer: Priority order for ties (default simpler-first).

    Returns:
        ``{layer_name: transform_name}``.
    """
    all_layers: set = set()
    for qsnr_dict in variant_qsnr.values():
        all_layers.update(qsnr_dict.keys())
    result: Dict[str, str] = {}
    tx_names = list(variant_qsnr.keys())

    def _preference_rank(tx: str) -> int:
        tx_lower = tx.lower()
        for i, candidate in enumerate(prefer):
            if candidate in tx_lower:
                return i
        return len(prefer)  # unknown transforms last

    for layer in all_layers:
        best_qsnr = max(
            variant_qsnr[tx].get(layer, -float("inf")) for tx in tx_names
        )
        candidates = [
            tx for tx in tx_names
            if variant_qsnr[tx].get(layer, -float("inf")) >= best_qsnr - eps
        ]
        candidates.sort(key=_preference_rank)
        result[layer] = candidates[0]
    return result
