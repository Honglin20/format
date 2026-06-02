"""Cross-config layer ranking: identify consistently worst layers across configs.

Usage::

    from src.report._study_report import StudyReport
    from src.analysis.cross_config_ranking import CrossConfigLayerRanking

    study_report = StudyReport.from_file(output_dir)
    ranking = CrossConfigLayerRanking.from_study(study_report)
    for layer, avg_qsnr in ranking.consistent_worst(k=5):
        print(f"  {layer}: avg QSNR = {avg_qsnr:.1f} dB")
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.report._study_report import StudyReport
    from src.session._result import SessionResult


class CrossConfigLayerRanking:
    """Compare which layers are consistently worst across multiple configs.

    Extracts per-config per-layer QSNR from a StudyReport and provides
    methods for finding consistently worst layers, config-specific worst,
    and cross-config deltas.
    """

    def __init__(self, data: Dict[str, Dict[str, float]]):
        """Initialize with {config_name: {layer_name: output_qsnr_db}}."""
        self._data = data
        self._all_layers: set = set()
        for config_map in data.values():
            self._all_layers.update(config_map.keys())

    @classmethod
    def from_study(cls, study_report: StudyReport) -> CrossConfigLayerRanking:
        """Extract per-config per-layer QSNR from a StudyReport."""
        data: Dict[str, Dict[str, float]] = {}
        for part_name, part_results in study_report._results.items():
            for r in part_results:
                if not r.qsnr_per_layer:
                    continue
                cfg_name = r.name or r.config.w_format if r.config else part_name
                layer_qsnr = {}
                for layer, qsnr in r.qsnr_per_layer.items():
                    if qsnr is not None and math.isfinite(qsnr):
                        layer_qsnr[layer] = qsnr
                data[cfg_name] = layer_qsnr
        return cls(data)

    @classmethod
    def from_results(cls, results: Dict[str, SessionResult]) -> CrossConfigLayerRanking:
        """Build from a dict of {config_name: SessionResult}."""
        data: Dict[str, Dict[str, float]] = {}
        for name, r in results.items():
            if not r.qsnr_per_layer:
                continue
            layer_qsnr = {}
            for layer, qsnr in r.qsnr_per_layer.items():
                if qsnr is not None and math.isfinite(qsnr):
                    layer_qsnr[layer] = qsnr
            data[name] = layer_qsnr
        return cls(data)

    @property
    def config_names(self) -> List[str]:
        return list(self._data.keys())

    @property
    def all_layers(self) -> set:
        return set(self._all_layers)

    def consistent_worst(self, k: int = 5) -> List[Tuple[str, float]]:
        """Layers that appear in worst-k across ALL configs.

        Returns [(layer_name, avg_qsnr)] sorted by avg QSNR ascending.
        """
        if not self._data:
            return []

        n_configs = len(self._data)
        # For each config, get the set of worst-k layers
        config_worst_sets: List[set] = []
        for cfg_name, layer_map in self._data.items():
            sorted_layers = sorted(layer_map.items(), key=lambda x: x[1])
            worst_set = {name for name, _ in sorted_layers[:k]}
            config_worst_sets.append(worst_set)

        # Intersection: layers in ALL worst-k sets
        if not config_worst_sets:
            return []
        consistent = config_worst_sets[0]
        for s in config_worst_sets[1:]:
            consistent = consistent & s

        if not consistent:
            return []

        # Compute avg QSNR across all configs for consistent layers
        avg_qsnr: Dict[str, float] = {}
        for layer in consistent:
            values = []
            for cfg_map in self._data.values():
                if layer in cfg_map:
                    values.append(cfg_map[layer])
            if values:
                avg_qsnr[layer] = sum(values) / len(values)

        return sorted(avg_qsnr.items(), key=lambda x: x[1])

    def config_specific_worst(self, config: str, k: int = 5) -> List[Tuple[str, float]]:
        """Layers worst in a specific config but NOT in the top-k of all configs."""
        if config not in self._data:
            return []

        config_map = self._data[config]
        sorted_layers = sorted(config_map.items(), key=lambda x: x[1])
        config_top_k = {name for name, _ in sorted_layers[:k]}

        # Layers in top-k of this config but not in top-k of all other configs
        other_worst = set()
        for cfg_name, cfg_map in self._data.items():
            if cfg_name == config:
                continue
            other_sorted = sorted(cfg_map.items(), key=lambda x: x[1])
            other_worst.update(name for name, _ in other_sorted[:k])

        specific = config_top_k - other_worst
        if not specific:
            return []

        result = [(name, config_map[name]) for name in specific if name in config_map]
        return sorted(result, key=lambda x: x[1])

    def layer_qsnr_delta(
        self, layer: str, from_config: str, to_config: str
    ) -> Optional[float]:
        """QSNR improvement for a layer between two configs.

        Returns to_config QSNR - from_config QSNR (positive = improvement).
        """
        from_q = self._data.get(from_config, {}).get(layer)
        to_q = self._data.get(to_config, {}).get(layer)
        if from_q is None or to_q is None:
            return None
        return to_q - from_q

    def role_dominance_cross_config(self, k: int = 5) -> List[dict]:
        """For worst-k layers, show role dominance per config.

        Returns [{layer, configs: [{config, dominant_role, qsnr}]}].
        Requires qsnr_by_role data in SessionResults.
        """
        # Get overall worst layers
        all_qsnr: Dict[str, List[float]] = {}
        for cfg_map in self._data.values():
            for layer, qsnr in cfg_map.items():
                all_qsnr.setdefault(layer, []).append(qsnr)

        avg_map = {l: sum(vs) / len(vs) for l, vs in all_qsnr.items()}
        worst_layers = sorted(avg_map.items(), key=lambda x: x[1])[:k]
        worst_names = [name for name, _ in worst_layers]

        result = []
        for layer in worst_names:
            entry = {"layer": layer, "configs": []}
            for cfg_name, cfg_map in self._data.items():
                qsnr = cfg_map.get(layer)
                if qsnr is not None:
                    entry["configs"].append({
                        "config": cfg_name,
                        "qsnr": round(qsnr, 1),
                    })
            result.append(entry)
        return result

    def summary(self, k: int = 5) -> str:
        """Human-readable summary table."""
        lines = [f"Cross-Config Layer Ranking ({len(self._data)} configs)"]

        consistent = self.consistent_worst(k)
        if consistent:
            lines.append(f"\nConsistently worst {k}:")
            for layer, avg_q in consistent:
                per_cfg = []
                for cfg_name, cfg_map in self._data.items():
                    q = cfg_map.get(layer)
                    per_cfg.append(f"{cfg_name}={q:.1f}" if q is not None else f"{cfg_name}=N/A")
                lines.append(f"  {layer}: avg={avg_q:.1f} dB  ({', '.join(per_cfg)})")
        else:
            lines.append(f"\nNo layers consistently in worst-{k} across all configs.")

        for cfg_name in self._data:
            specific = self.config_specific_worst(cfg_name, k)
            if specific:
                lines.append(f"\n{cfg_name}-specific worst:")
                for layer, qsnr in specific[:3]:
                    lines.append(f"  {layer}: {qsnr:.1f} dB")

        return "\n".join(lines)
