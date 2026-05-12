"""ErrorProvenance — systematic per-role per-layer error attribution.

Answers "where does the error come from?" by combining multi-role QSNR/MSE
data with the existing error-source (Source/Mixed/Propagated) analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.session._result import SessionResult

# Roles we care about for diagnosis.
_DIAG_ROLES = ("input", "weight", "output")

# Layer type inference: map module name to a short type label.
_LAYER_TYPE_PATTERNS = [
    ("linear", "Linear"),
    ("fc", "Linear"),
    ("conv", "Conv"),
    ("norm", "Norm"),
    ("bn", "Norm"),
    ("layernorm", "Norm"),
    ("rmsnorm", "Norm"),
    ("activation", "Activation"),
    ("relu", "Activation"),
    ("gelu", "Activation"),
    ("softmax", "Softmax"),
    ("pool", "Pool"),
    ("embed", "Embed"),
]


def _infer_layer_type(layer_name: str) -> str:
    lower = layer_name.lower()
    for pat, label in _LAYER_TYPE_PATTERNS:
        if pat in lower:
            return label
    return "Other"


def _select_dominant(qsnr_by_role: dict, layers: list) -> dict:
    """For each layer, pick the role with the lowest QSNR.

    Returns ``{layer: dominant_role}``.
    """
    dominant = {}
    for layer in layers:
        best_role = "?"
        best_qsnr = float("inf")
        for role in _DIAG_ROLES:
            v = qsnr_by_role.get(role, {}).get(layer)
            if v is not None and v < best_qsnr:
                best_qsnr = v
                best_role = role
        dominant[layer] = best_role
    return dominant


class ErrorProvenance:
    """Per-role, per-layer error attribution on a single SessionResult.

    Usage::

        prov = result.diagnose
        print(prov.summary())
        print(prov.per_role_table())
        print(prov.error_source_analysis())
        for name, qsnr in prov.top_k(5, role="weight"):
            ...
    """

    def __init__(self, result: SessionResult):
        self._result = result

    # ------------------------------------------------------------------
    # summary: role × layer_type aggregation
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Aggregate QSNR statistics grouped by (role, layer_type).

        Returns:
            Formatted text table.
        """
        qsnr_by_role = self._result.qsnr_by_role
        if not qsnr_by_role:
            return "(No per-role QSNR data available. Run session.analyze() with QSNRObserver.)"

        # Collect per (role, ltype) values
        groups: dict = defaultdict(list)
        for role in sorted(qsnr_by_role):
            for layer, v in qsnr_by_role[role].items():
                if v is not None and v == v and v != float("inf") and v != float("-inf"):
                    ltype = _infer_layer_type(layer)
                    groups[(role, ltype)].append(v)

        if not groups:
            return "(No finite QSNR values found.)"

        lines = []
        hdr = f"{'Role':<10} {'Type':<12} {'Count':>7} {'Avg QSNR':>10} {'Min QSNR':>10} {'Std':>8}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for (role, ltype), values in sorted(groups.items()):
            if not values:
                continue
            avg = sum(values) / len(values)
            mn = min(values)
            # Sample std
            if len(values) > 1:
                ss = sum((v - avg) ** 2 for v in values)
                std = math.sqrt(ss / (len(values) - 1))
            else:
                std = 0.0
            lines.append(
                f"{role:<10} {ltype:<12} {len(values):>7} "
                f"{avg:>10.1f} {mn:>10.1f} {std:>8.1f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # per_role_table: per-layer input / weight / output columns
    # ------------------------------------------------------------------

    def per_role_table(self, max_layers: int = 40) -> str:
        """Per-layer table with input / weight / output QSNR columns.

        Sorted by worst (lowest) QSNR across all roles first.

        Args:
            max_layers: Maximum number of layers to display.

        Returns:
            Formatted text table.
        """
        qsnr_by_role = self._result.qsnr_by_role
        if not qsnr_by_role:
            return "(No per-role QSNR data available.)"

        # Union of all layers that appear in any role
        all_layers: set = set()
        for role_map in qsnr_by_role.values():
            all_layers.update(role_map.keys())

        # Compute worst QSNR per layer for sorting
        layer_worst: dict = {}
        for layer in all_layers:
            worst = float("inf")
            for role in _DIAG_ROLES:
                v = qsnr_by_role.get(role, {}).get(layer)
                if v is not None and v < worst:
                    worst = v
            layer_worst[layer] = worst

        ordered = sorted(all_layers, key=lambda n: layer_worst.get(n, float("inf")))

        dominant = _select_dominant(qsnr_by_role, ordered)

        lines = [
            f"{'Layer':<30} {'Input':>10} {'Weight':>10} {'Output':>10}  Dominant"
        ]
        lines.append("-" * len(lines[0]))
        for layer in ordered[:max_layers]:
            inv = qsnr_by_role.get("input", {}).get(layer)
            wv = qsnr_by_role.get("weight", {}).get(layer)
            ov = qsnr_by_role.get("output", {}).get(layer)
            i_str = f"{inv:.1f}" if inv is not None else "N/A"
            w_str = f"{wv:.1f}" if wv is not None else "N/A"
            o_str = f"{ov:.1f}" if ov is not None else "N/A"
            dom = dominant.get(layer, "?")
            lines.append(
                f"{layer:<30} {i_str:>10} {w_str:>10} {o_str:>10}  {dom}"
            )

        if len(ordered) > max_layers:
            lines.append(f"  ... and {len(ordered) - max_layers} more layers")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # top_k
    # ------------------------------------------------------------------

    def top_k(
        self,
        k: int = 10,
        role: str = "output",
    ) -> List[Tuple[str, float]]:
        """Return the *k* layers with the lowest QSNR.

        Args:
            k: Number of layers to return.
            role: Tensor role (``"input"`` / ``"weight"`` / ``"output"``).
                When ``"auto"``, each layer's worst role is used for sorting,
                and the returned tuples contain ``(layer, qsnr_of_worst_role)``.

        Returns:
            List of ``(layer_name, qsnr_db)`` sorted ascending (worst first).
        """
        if role == "auto":
            qsnr_by_role = self._result.qsnr_by_role
            if not qsnr_by_role:
                return []
            worst: dict = {}
            for layer in set().union(*(m.keys() for m in qsnr_by_role.values())):
                worst_q = float("inf")
                for r in _DIAG_ROLES:
                    v = qsnr_by_role.get(r, {}).get(layer)
                    if v is not None and v < worst_q:
                        worst_q = v
                if worst_q != float("inf"):
                    worst[layer] = worst_q
            return sorted(worst.items(), key=lambda x: x[1])[:k]

        role_map = self._result.qsnr_by_role.get(role, {})
        if not role_map:
            return []
        sorted_layers = sorted(
            [(n, v) for n, v in role_map.items() if v is not None and v == v],
            key=lambda x: x[1],
        )
        return sorted_layers[:k]

    # ------------------------------------------------------------------
    # error_source_analysis (delegates to existing logic)
    # ------------------------------------------------------------------

    def error_source_analysis(self, role: str = "output") -> str:
        """Per-layer error source diagnosis (Source / Mixed / Propagated).

        Delegates to ``SessionResult.tables.error_source_analysis()``.
        """
        return self._result.tables.error_source_analysis(role=role)

    # ------------------------------------------------------------------
    # depth_decay_data — for plotting (used by SessionPlotAccessor)
    # ------------------------------------------------------------------

    def depth_decay_data(self, role: str = "output") -> List[Tuple[int, str, float]]:
        """Return ``(depth_index, layer_name, qsnr)`` tuples for a QSNR-vs-depth plot.

        Layers are ordered by their position in the model (as encountered during
        ``named_modules()``).  Layers not present in the QSNR data are skipped.
        """
        role_map = self._result.qsnr_by_role.get(role, {})
        if not role_map:
            # Try accum if local is empty
            role_map = self._result.accum_qsnr_per_layer
        if not role_map:
            return []

        points = []
        for idx, layer in enumerate(role_map):
            v = role_map[layer]
            if v == v and v != float("inf") and v != float("-inf"):
                points.append((idx, layer, v))
        return points
