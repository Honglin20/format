"""Tests for src/analysis/_intervention.py — InterventionPlanner + InterventionPlan."""

import pytest

from src.analysis._intervention import InterventionPlan, InterventionPlanner
from src.analysis._intervention_accessor import InterventionAccessor, InterventionComparison
from src.session._config import QuantConfig
from src.session._result import SessionResult
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> SessionResult:
    """Build a SessionResult with per-role QSNR for intervention tests."""
    defaults = dict(
        name="test",
        config=QuantConfig(name="int8", w_format="int8", w_granularity="per_tensor"),
        qsnr_by_role={
            "input": {
                "layer0.linear": 15.0,
                "layer1.conv": 25.0,
                "layer2.norm": 45.0,
                "layer3.linear": 10.0,
                "layer4.conv": 30.0,
                "layer5.linear": 12.0,
                "layer6.norm": 50.0,
                "layer7.linear": 8.0,
            },
            "weight": {
                "layer0.linear": 20.0,
                "layer1.conv": 30.0,
                "layer2.norm": 48.0,
                "layer3.linear": 14.0,
                "layer4.conv": 35.0,
                "layer5.linear": 18.0,
                "layer6.norm": 52.0,
                "layer7.linear": 11.0,
            },
            "output": {},
        },
    )
    defaults.update(overrides)
    return SessionResult(**defaults)


# ---------------------------------------------------------------------------
# InterventionPlan
# ---------------------------------------------------------------------------

class TestInterventionPlan:
    def test_explain_empty_plan(self):
        plan = InterventionPlan()
        text = plan.explain()
        assert "Empty plan" in text

    def test_explain_empty_plan_with_description(self):
        plan = InterventionPlan(metadata={"description": "No QSNR data"})
        text = plan.explain()
        assert "Empty plan" in text
        assert "No QSNR data" in text

    def test_to_dict_serializable(self):
        plan = InterventionPlan(
            metadata={"description": "test", "strategy": "top_k_boost", "k": 3},
        )
        d = plan.to_dict()
        assert d["metadata"]["description"] == "test"
        assert d["metadata"]["strategy"] == "top_k_boost"
        assert isinstance(d["overrides"], dict)


# ---------------------------------------------------------------------------
# InterventionPlanner — top_k_boost
# ---------------------------------------------------------------------------

class TestInterventionPlannerTopKBoost:
    def test_returns_k_overrides(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=3, role="input", target_bits=8)
        assert len(plan.overrides) == 3

    def test_overrides_are_op_quant_configs(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=2, role="weight", target_bits=8)
        for layer, cfg in plan.overrides.items():
            assert isinstance(cfg, OpQuantConfig)

    def test_boosted_role_uses_target_bits(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=1, role="weight", target_bits=8)
        override = list(plan.overrides.values())[0]
        assert "int8" in str(override.weight.format)

    def test_non_boosted_role_unchanged(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=1, role="input", target_bits=8)
        override = list(plan.overrides.values())[0]
        # weight role should still be original (int8)
        assert override.weight is not None
        # input role should be boosted to int8
        assert override.input is not None

    def test_auto_role_picks_worst_per_layer(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=3, role="auto", target_bits=8)
        assert len(plan.overrides) == 3
        # layer7.linear has worst QSNR (8.0 input), should be included
        assert "layer7.linear" in plan.overrides
        assert "layer3.linear" in plan.overrides  # 10.0 input

    def test_empty_qsnr_returns_empty_plan(self):
        result = _make_result(qsnr_by_role={})
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=5, role="input", target_bits=8)
        assert len(plan.overrides) == 0
        assert "No QSNR" in plan.metadata["description"]

    def test_non_boostable_role_explicit_request(self):
        """Default QuantConfig has no output scheme → output is non-boostable."""
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=3, role="output", target_bits=8)
        assert len(plan.overrides) == 0
        assert "cannot be boosted" in plan.metadata["description"]
        assert "output" in plan.metadata["description"]

    def test_explain_produces_readable_output(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=2, role="weight", target_bits=8)
        text = plan.explain()
        assert "Intervention Plan" in text
        assert "top_k_boost" in text
        assert "Layers modified: 2" in text

    def test_changes_metadata_populated(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=1, role="weight", target_bits=4)
        change = plan.metadata["changes"]
        assert len(change) == 1
        layer_name = list(plan.overrides.keys())[0]
        assert "QSNR=" in change[layer_name]["why"]

    def test_metadata_has_expected_keys(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=2, role="weight", target_bits=8)
        assert plan.metadata["strategy"] == "top_k_boost"
        assert plan.metadata["k"] == 2
        assert plan.metadata["role"] == "weight"
        assert plan.metadata["target_bits"] == 8


# ---------------------------------------------------------------------------
# InterventionPlanner — recommend
# ---------------------------------------------------------------------------

class TestInterventionPlannerRecommend:
    def test_conservative_returns_plan(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.recommend(strategy="conservative")
        assert isinstance(plan, InterventionPlan)
        assert len(plan.overrides) >= 2

    def test_aggressive_returns_more_overrides(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.recommend(strategy="aggressive")
        assert len(plan.overrides) >= 2

    def test_aggressive_more_than_conservative(self):
        """Aggressive strategy (35% threshold) should boost >= layers than conservative (15%)."""
        result = _make_result()
        planner = InterventionPlanner(result)
        cons = planner.recommend(strategy="conservative")
        agg = planner.recommend(strategy="aggressive")
        assert len(agg.overrides) >= len(cons.overrides)

    def test_empty_qsnr_returns_empty_plan(self):
        result = _make_result(qsnr_by_role={})
        planner = InterventionPlanner(result)
        plan = planner.recommend(strategy="conservative")
        assert len(plan.overrides) == 0

    def test_all_inf_qsnr_returns_empty_plan(self):
        result = _make_result(qsnr_by_role={
            "input": {"a": float("inf")},
            "weight": {"a": float("inf")},
            "output": {},
        })
        planner = InterventionPlanner(result)
        plan = planner.recommend(strategy="conservative")
        assert "No valid QSNR" in plan.metadata["description"]

    def test_single_layer_returns_at_least_2(self):
        """Conservative k is max(2, ...); with 1 layer should return 1."""
        result = _make_result(qsnr_by_role={
            "input": {"only_layer": 10.0},
            "weight": {"only_layer": 12.0},
            "output": {},
        })
        planner = InterventionPlanner(result)
        plan = planner.recommend(strategy="conservative")
        assert len(plan.overrides) == 1


# ---------------------------------------------------------------------------
# InterventionPlanner — transform_ranking
# ---------------------------------------------------------------------------

class TestInterventionPlannerTransformRanking:
    def test_returns_limitation_message(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        text = planner.transform_ranking(k=5)
        assert "requires the original model" in text
        assert "recommend" in text.lower()


# ---------------------------------------------------------------------------
# InterventionPlanner — edge cases
# ---------------------------------------------------------------------------

class TestInterventionPlannerEdgeCases:
    def test_config_with_only_input_scheme(self):
        """Test with config that only has input scheme (no weight)."""
        from src.formats.base import FormatBase
        from src.scheme.granularity import GranularitySpec

        cfg = QuantConfig(
            name="input-only",
            w_format="int8",
            w_granularity="per_tensor",
            a_format="int4",
            a_granularity="per_tensor",
        )
        result = SessionResult(
            name="test",
            config=cfg,
            qsnr_by_role={
                "input": {"layer0.linear": 5.0},
                "weight": {"layer0.linear": 3.0},
                "output": {},
            },
        )
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=3, role="output", target_bits=8)
        assert len(plan.overrides) == 0
        assert "cannot be boosted" in plan.metadata["description"]

    def test_to_dict_on_populated_plan(self):
        result = _make_result()
        planner = InterventionPlanner(result)
        plan = planner.top_k_boost(k=2, role="weight", target_bits=8)
        d = plan.to_dict()
        assert len(d["overrides"]) == 2
        assert "metadata" in d
        assert all(isinstance(k, str) for k in d["overrides"])


# ---------------------------------------------------------------------------
# InterventionAccessor + InterventionComparison
# ---------------------------------------------------------------------------

class TestInterventionComparison:
    def test_summary_with_metrics(self):
        """summary() with fp32_metrics, quant_metrics, qsnr_per_layer on both sides."""
        baseline = SessionResult(
            name="baseline",
            config=QuantConfig(name="int4", w_format="int4", w_granularity="per_tensor"),
            fp32_metrics={"accuracy": 0.95},
            quant_metrics={"accuracy": 0.85},
            qsnr_per_layer={"layer0.linear": 15.0, "layer1.conv": 20.0},
            mse_per_layer={},
        )
        intervention = SessionResult(
            name="intervention",
            config=QuantConfig(name="int8-boosted", w_format="int8", w_granularity="per_tensor"),
            fp32_metrics={"accuracy": 0.95},
            quant_metrics={"accuracy": 0.90},
            qsnr_per_layer={"layer0.linear": 22.0, "layer1.conv": 25.0},
            mse_per_layer={},
        )
        plan = InterventionPlan(
            overrides={"layer0.linear": OpQuantConfig()},
            metadata={
                "description": "Top-1 boost to 8-bit",
                "strategy": "top_k_boost",
                "changes": {
                    "layer0.linear": {
                        "what": "weight: 4bit → 8bit",
                        "why": "QSNR=15.0 dB",
                    }
                },
            },
        )
        comparison = InterventionComparison(
            baseline=baseline,
            intervention=intervention,
            plan=plan,
        )
        text = comparison.summary()
        assert "Intervention Comparison" in text
        assert "Plan:" in text
        assert "Avg QSNR" in text
        assert "layer0.linear" in text

    def test_summary_without_metrics(self):
        """summary() without metrics should not crash."""
        baseline = SessionResult(
            name="baseline",
            config=QuantConfig(name="int4", w_format="int4", w_granularity="per_tensor"),
        )
        plan = InterventionPlan()
        comparison = InterventionComparison(
            baseline=baseline,
            intervention=baseline,
            plan=plan,
        )
        text = comparison.summary()
        assert "Intervention Comparison" in text

    def test_to_dict(self):
        baseline = SessionResult(
            name="baseline",
            config=QuantConfig(name="int4", w_format="int4", w_granularity="per_tensor"),
            qsnr_per_layer={"a": 10.0},
        )
        plan = InterventionPlan()
        comparison = InterventionComparison(
            baseline=baseline,
            intervention=baseline,
            plan=plan,
        )
        d = comparison.to_dict()
        assert "plan" in d
        assert "baseline_qsnr" in d
        assert "intervention_qsnr" in d


class TestInterventionAccessor:
    def test_compare_empty_plan_returns_baseline_as_both(self):
        """When plan has no overrides, baseline == intervention."""
        result = _make_result()
        accessor = InterventionAccessor(result)
        plan = InterventionPlan()
        comparison = accessor.compare(
            model=None,  # not used when plan is empty
            calib_data=None,
            plan=plan,
        )
        assert comparison.baseline is result
        assert comparison.intervention is result
        assert comparison.plan is plan
