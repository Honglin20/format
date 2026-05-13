"""InterventionPlanner — generate per-layer precision-boost and transform plans.

Consumes ``SessionResult`` data (QSNR by role) and produces
``InterventionPlan`` objects that can be fed to a new ``Session`` via the
``overrides`` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session._result import SessionResult
    from src.scheme.op_config import OpQuantConfig

# Roles considered for intervention.
_DIAG_ROLES = ("input", "weight", "output")


# ── InterventionPlan ─────────────────────────────────────────────────────────


@dataclass
class InterventionPlan:
    """A set of per-layer config overrides generated from a diagnostic analysis.

    Attributes:
        overrides: ``{layer_name: OpQuantConfig}`` — ready to pass to
            ``Session(..., overrides=plan.overrides)``.
        metadata: Human-readable metadata about the plan (strategy, k, role).
    """

    overrides: Dict[str, "OpQuantConfig"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        """Human-readable table explaining each override.

        Returns:
            Formatted text showing layer, what was changed, and why.
        """
        if not self.overrides:
            return "(Empty plan — no overrides.)"

        lines = [
            f"Intervention Plan: {self.metadata.get('description', '(unnamed)')}",
            f"  Strategy: {self.metadata.get('strategy', 'manual')}",
            f"  Layers modified: {len(self.overrides)}",
            "",
            f"{'Layer':<30} {'Change':<40} {'Reason':<30}",
            "-" * 100,
        ]
        changes = self.metadata.get("changes", {})
        for layer in sorted(self.overrides):
            change_info = changes.get(layer, {})
            what = change_info.get("what", "config override")
            why = change_info.get("why", "")
            lines.append(f"{layer:<30} {what:<40} {why:<30}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the plan to a JSON-friendly dict.

        The overrides themselves (``OpQuantConfig``) are stored as string
        representations.  For full round-trip serialisation use the
        ``SessionResult`` persistence.
        """
        return {
            "metadata": dict(self.metadata),
            "overrides": {k: repr(v) for k, v in self.overrides.items()},
        }


# ── InterventionPlanner ─────────────────────────────────────────────────────


class InterventionPlanner:
    """Generate intervention plans from a single SessionResult.

    Usage::

        planner = result.plan
        plan = planner.top_k_boost(k=5, role="weight", target_bits=8)
        plan = planner.recommend(strategy="conservative")
    """

    def __init__(self, result: SessionResult):
        self._result = result

    # ── top_k_boost ─────────────────────────────────────────────────────

    def top_k_boost(
        self,
        k: int = 5,
        role: str = "auto",
        target_bits: int = 8,
    ) -> InterventionPlan:
        """Boost the *k* worst-QSNR layers by raising bit-width for *role*.

        Args:
            k: Number of layers to boost.
            role: Tensor role to boost (``"input"`` / ``"weight"`` /
                ``"output"``), or ``"auto"`` to pick the worst role per layer.
            target_bits: Target bit-width (e.g. 8 for INT8).

        Returns:
            ``InterventionPlan`` with per-layer ``OpQuantConfig`` overrides.
        """
        qsnr_by_role = self._result.qsnr_by_role
        if not qsnr_by_role:
            return InterventionPlan(
                metadata={"description": "No QSNR data — empty plan"},
            )

        # Determine which roles are boostable (have non-None schemes in base cfg)
        base_cfg = self._result.config.to_op_config() if self._result.config else None
        boostable_roles = set()
        if base_cfg is not None:
            for r in _DIAG_ROLES:
                if getattr(base_cfg, r, None) is not None:
                    boostable_roles.add(r)

        # Build candidate list: (layer, role, qsnr)
        # Skip roles that can't be boosted (None scheme in base config).
        candidates = []
        for r in (role,) if role != "auto" else tuple(boostable_roles):
            if r not in qsnr_by_role:
                continue
            for layer, qsnr in qsnr_by_role[r].items():
                if qsnr is not None and qsnr == qsnr:
                    candidates.append((layer, r, qsnr))

        if role == "auto":
            # Keep only the worst role per layer
            best_per_layer: dict = {}
            for layer, r, qsnr in candidates:
                if layer not in best_per_layer or qsnr < best_per_layer[layer][1]:
                    best_per_layer[layer] = (r, qsnr)
            candidates = [(l, r, q) for l, (r, q) in best_per_layer.items()]

        # Sort by QSNR ascending (worst first)
        candidates.sort(key=lambda x: x[2])

        # Take top k
        selected = candidates[:k]

        overrides, changes = self._build_bit_boost_overrides(selected, target_bits)

        role_label = role if role != "auto" else "auto (worst per layer)"
        return InterventionPlan(
            overrides=overrides,
            metadata={
                "description": f"Top-{k} {role_label} boost to {target_bits}-bit",
                "strategy": "top_k_boost",
                "k": k,
                "role": role,
                "target_bits": target_bits,
                "changes": changes,
            },
        )

    def _build_bit_boost_overrides(
        self, selected: list, target_bits: int
    ) -> tuple:
        """Build OpQuantConfig overrides for a list of (layer, role, qsnr)."""
        from src.formats.base import FormatBase
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme

        overrides = {}
        changes = {}

        for layer, role, qsnr in selected:
            # Build overrides for all roles of this layer (base from result config)
            base_cfg = self._result.config.to_op_config() if self._result.config else None
            if base_cfg is None:
                continue

            # Skip roles whose scheme is None in the base config — boosting
            # them would be a no-op (e.g. output / bias are typically None).
            current_scheme = getattr(base_cfg, role, None)
            if current_scheme is None:
                continue

            target_fmt = f"int{target_bits}"
            fmt = current_scheme.format
            current_bits = fmt.mbits if fmt.ebits == 0 else 1 + fmt.ebits + fmt.mbits

            new_cfg_dict = {}
            for field_name in base_cfg.__dataclass_fields__:
                scheme = getattr(base_cfg, field_name)
                if scheme is not None and field_name in ("input", "weight", "output"):
                    if field_name == role:
                        new_fmt = FormatBase.from_str(target_fmt)
                        new_cfg_dict[field_name] = QuantScheme(
                            format=new_fmt,
                            granularity=scheme.granularity,
                            transform=scheme.transform,
                            round_mode=scheme.round_mode,
                            scale_storage=scheme.scale_storage,
                        )
                    else:
                        new_cfg_dict[field_name] = scheme
                else:
                    new_cfg_dict[field_name] = scheme

            overrides[layer] = OpQuantConfig(**new_cfg_dict)

            bits_from = f"{current_bits}bit" if current_bits else str(current_scheme.format)
            changes[layer] = {
                "what": f"{role}: {bits_from} → {target_bits}bit",
                "why": f"QSNR={qsnr:.1f} dB (worst {role})",
            }

        return overrides, changes

    # ── transform_ranking ───────────────────────────────────────────────

    def transform_ranking(self, k: int = 10) -> str:
        """Rank transform candidates (none / hadamard / smoothquant) for the
        *k* worst-QSNR layers by estimated matmul-output QSNR.

        This requires the original model and calibration data, which are not
        stored in SessionResult.  Use :meth:`recommend` for an end-to-end
        workflow that runs a helper Session.

        Returns:
            Formatted text table, or a message indicating the limitation.
        """
        return (
            "transform_ranking requires the original model and calibration data.\n"
            "Use planner.recommend(strategy='conservative') for the full workflow\n"
            "that evaluates transforms automatically."
        )

    # ── recommend ───────────────────────────────────────────────────────

    def recommend(self, strategy: str = "conservative") -> InterventionPlan:
        """Generate a combined recommendation based on QSNR ranking.

        Currently implements pure QSNR-based top-k overrides.  Transform
        selection and distribution-guided recommendations will be added
        in a follow-up.

        Args:
            strategy: ``"conservative"`` — few layers, large margin.
                      ``"aggressive"`` — more layers, lower threshold.

        Returns:
            ``InterventionPlan``.
        """
        qsnr_by_role = self._result.qsnr_by_role
        if not qsnr_by_role:
            return InterventionPlan(
                metadata={"description": "No QSNR data — empty plan"},
            )

        # Determine threshold and k based on strategy
        all_qsnr = []
        for role in _DIAG_ROLES:
            for v in qsnr_by_role.get(role, {}).values():
                if v is not None and v == v and v != float("-inf"):
                    all_qsnr.append(v)

        if not all_qsnr:
            return InterventionPlan(
                metadata={"description": "No valid QSNR data"},
            )

        all_qsnr.sort()

        # Count unique layers (per-role values may be 3x layer count)
        n_layers = len(set().union(*(m.keys() for m in qsnr_by_role.values()))) if qsnr_by_role else 0

        if strategy == "conservative":
            threshold = all_qsnr[0] + (all_qsnr[-1] - all_qsnr[0]) * 0.15
            k = max(2, sum(1 for v in all_qsnr if v < threshold))
        else:  # aggressive
            threshold = all_qsnr[0] + (all_qsnr[-1] - all_qsnr[0]) * 0.35
            k = max(4, sum(1 for v in all_qsnr if v < threshold))

        return self.top_k_boost(k=min(k, n_layers), role="auto", target_bits=8)
