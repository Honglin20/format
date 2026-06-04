"""Tests for src/api/diagnostic_api.py — coarse_pass + deep_dive + prescribe."""

import json
import math
import os

import pytest
import torch

from src.analysis.observers import (
    PerBlockQSNRObserver, QSNRObserver, DistributionObserver,
)
from src.observer.events import QuantEvent
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.formats.base import FormatBase
from src.session import QuantConfig
from src.session._result import SessionResult
from src.api.diagnostic_api import _safe_layer_name


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_event(fp32, quant, layer="fc1", role="weight",
                mode=GranularityMode.PER_BLOCK, block_size=16):
    if mode == GranularityMode.PER_TENSOR:
        gran = GranularitySpec(mode=mode)
    else:
        gran = GranularitySpec(mode=mode, block_size=block_size, block_axis=-1)
    scheme = QuantScheme(format=FormatBase.from_str("int8"), granularity=gran)
    return QuantEvent(
        layer_name=layer, role=role, stage="quantize", pipeline_index=0,
        fp32_tensor=fp32, quant_tensor=quant, scheme=scheme,
    )


def _make_result(*, name="test", accuracy=None, fp32_accuracy=None,
                 qsnr_per_layer=None, qsnr_by_role=None,
                 accum_qsnr=None, observers_data=None):
    config = QuantConfig(
        name=name, w_format="int8", w_granularity="per_block",
        w_block_size=16, a_format="int8", a_granularity="per_block",
        a_block_size=16,
    )
    quant_metrics = {"accuracy": accuracy} if accuracy is not None else {}
    fp32_metrics = {"accuracy": fp32_accuracy} if fp32_accuracy is not None else {}
    return SessionResult(
        name=name,
        config=config,
        fp32_metrics=fp32_metrics,
        quant_metrics=quant_metrics,
        qsnr_per_layer=qsnr_per_layer or {},
        mse_per_layer={},
        qsnr_by_role=qsnr_by_role or {},
        mse_by_role={},
        accum_qsnr_per_layer=accum_qsnr or {},
        accum_mse_per_layer={},
        observers_data=observers_data or {},
    )


def _build_result_with_observers(
    name="test",
    layers=("fc1", "fc2"),
    accuracy=0.85,
    fp32_accuracy=0.92,
):
    """Build a SessionResult with QSNR + Distribution + PerBlock observers."""
    obs_qsnr = QSNRObserver()
    obs_dist = DistributionObserver()
    obs_block = PerBlockQSNRObserver()

    qsnr_per_layer = {}
    qsnr_by_role = {"input": {}, "weight": {}, "output": {}}
    accum_qsnr = {}
    obs_data = {}

    for i, layer in enumerate(layers):
        for role in ("weight", "input"):
            fp32 = torch.randn(64)
            quant = fp32 + 0.01 * torch.randn(64)

            # Per-block event
            event_block = _make_event(fp32, quant, layer=layer, role=role, block_size=4)
            obs_block.on_event(event_block)
            obs_qsnr.on_event(event_block)

            # Per-tensor event for distribution
            event_pt = _make_event(
                fp32, quant, layer=layer, role=role,
                mode=GranularityMode.PER_TENSOR,
            )
            obs_dist.on_event(event_pt)

        # Assign QSNR values (simulate decreasing quality for deeper layers)
        base_qsnr = 35.0 - i * 5.0
        qsnr_per_layer[layer] = base_qsnr
        accum_qsnr[layer] = base_qsnr - 2.0
        qsnr_by_role["input"][layer] = base_qsnr - 3.0
        qsnr_by_role["weight"][layer] = base_qsnr - 1.0
        qsnr_by_role["output"][layer] = base_qsnr

    # Merge observer data
    for obs in (obs_block, obs_dist, obs_qsnr):
        for layer, roles in obs.report().items():
            obs_data.setdefault(layer, {})
            for role, stages in roles.items():
                obs_data[layer].setdefault(role, {})
                for stage, slices in stages.items():
                    obs_data[layer][role].setdefault(stage, {})
                    obs_data[layer][role][stage].update(slices)

    return _make_result(
        name=name,
        accuracy=accuracy,
        fp32_accuracy=fp32_accuracy,
        qsnr_per_layer=qsnr_per_layer,
        qsnr_by_role=qsnr_by_role,
        accum_qsnr=accum_qsnr,
        observers_data=obs_data,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# coarse_pass tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoarsePass:

    def test_empty_results(self):
        from src.api.diagnostic_api import coarse_pass
        report = coarse_pass({})
        assert report.fp32_accuracy is None
        assert report.gaps == []

    def test_single_config(self):
        from src.api.diagnostic_api import coarse_pass
        r = _make_result(name="W8A8", accuracy=0.90, fp32_accuracy=0.92)
        report = coarse_pass({"W8A8": r})
        assert report.fp32_accuracy == 0.92
        assert len(report.gaps) == 1
        assert report.gaps[0].config == "W8A8"
        assert report.gaps[0].accuracy == 0.90
        assert report.gaps[0].delta_from_fp32 == pytest.approx(-0.02)

    def test_multiple_configs_with_bottleneck(self):
        from src.api.diagnostic_api import coarse_pass
        results = {
            "W8A8": _make_result(name="W8A8", accuracy=0.90, fp32_accuracy=0.92,
                                 qsnr_per_layer={"fc1": 40.0, "fc2": 38.0}),
            "W4A8": _make_result(name="W4A8", accuracy=0.82, fp32_accuracy=0.92,
                                 qsnr_per_layer={"fc1": 30.0, "fc2": 28.0}),
            "W4A4": _make_result(name="W4A4", accuracy=0.75, fp32_accuracy=0.92,
                                 qsnr_per_layer={"fc1": 22.0, "fc2": 20.0}),
        }
        report = coarse_pass(results)
        assert report.bottleneck.weight_degradation == pytest.approx(0.08)
        assert report.bottleneck.activation_degradation == pytest.approx(0.07)
        assert report.bottleneck.primary in ("weight", "both")

    def test_cross_config_ranking(self):
        from src.api.diagnostic_api import coarse_pass
        results = {
            "int8": _make_result(name="int8",
                                 qsnr_per_layer={"fc1": 35.0, "fc2": 30.0, "fc3": 40.0}),
            "int4": _make_result(name="int4",
                                 qsnr_per_layer={"fc1": 25.0, "fc2": 18.0, "fc3": 35.0}),
        }
        report = coarse_pass(results, k=3)
        # fc2 should be consistently worst
        worst_names = [r.layer for r in report.consistent_worst]
        assert "fc2" in worst_names

    def test_transform_effects_detected(self):
        from src.api.diagnostic_api import coarse_pass
        results = {
            "W4A4": _make_result(name="W4A4", accuracy=0.75, fp32_accuracy=0.92),
            "W4A4+SQ": _make_result(name="W4A4+SQ", accuracy=0.85, fp32_accuracy=0.92),
        }
        report = coarse_pass(results)
        assert len(report.transform_effects) == 1
        assert report.transform_effects[0].transform == "smoothquant"
        assert report.transform_effects[0].accuracy_gain == pytest.approx(0.10)

    def test_transform_hadamard(self):
        from src.api.diagnostic_api import coarse_pass
        results = {
            "W4A4": _make_result(name="W4A4", accuracy=0.75, fp32_accuracy=0.92),
            "W4A4+HD": _make_result(name="W4A4+HD", accuracy=0.83, fp32_accuracy=0.92),
        }
        report = coarse_pass(results)
        assert len(report.transform_effects) == 1
        assert report.transform_effects[0].transform == "hadamard"

    def test_summary_method(self):
        from src.api.diagnostic_api import coarse_pass
        results = {
            "W8A8": _make_result(name="W8A8", accuracy=0.90, fp32_accuracy=0.92),
        }
        report = coarse_pass(results)
        text = report.summary()
        assert "Coarse Analysis" in text
        assert "W8A8" in text

    def test_to_dict(self):
        from src.api.diagnostic_api import coarse_pass
        results = {"cfg": _make_result(name="cfg", accuracy=0.85, fp32_accuracy=0.92)}
        report = coarse_pass(results)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "gaps" in d
        assert "bottleneck" in d
        assert d["fp32_accuracy"] == 0.92

    def test_fp32_accuracy_from_explicit(self):
        from src.api.diagnostic_api import coarse_pass
        r = _make_result(name="cfg", accuracy=0.85)
        report = coarse_pass({"cfg": r}, fp32_accuracy=0.95)
        assert report.fp32_accuracy == 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# deep_dive tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepDive:

    def test_empty_result(self):
        from src.api.diagnostic_api import deep_dive
        report = deep_dive(_make_result())
        assert report.layer_diagnoses == []
        assert report.block_analyses == []

    def test_with_observers_data(self):
        from src.api.diagnostic_api import deep_dive
        result = _build_result_with_observers(layers=("fc1", "fc2"))
        report = deep_dive(result, layers=["fc1"])

        # Should have distribution diagnoses for fc1
        assert len(report.layer_diagnoses) > 0
        fc1_diag = [d for d in report.layer_diagnoses if d.layer == "fc1"]
        assert len(fc1_diag) > 0
        assert fc1_diag[0].classification != ""
        assert fc1_diag[0].qsnr_db > 0

    def test_block_analyses(self):
        from src.api.diagnostic_api import deep_dive
        result = _build_result_with_observers(layers=("fc1",))
        report = deep_dive(result, layers=["fc1"])

        assert len(report.block_analyses) > 0
        block = report.block_analyses[0]
        assert block.layer == "fc1"
        assert block.unit_type in ("block", "channel")
        assert "mean" in block.stats
        assert len(block.worst_units) > 0

    def test_depth_decay(self):
        from src.api.diagnostic_api import deep_dive
        result = _build_result_with_observers(layers=("fc1", "fc2", "fc3"))
        report = deep_dive(result)

        if report.depth_decay:
            assert len(report.depth_decay) > 0
            assert report.depth_decay[0].depth == 0
            assert report.depth_decay[0].qsnr_db > 0

    def test_error_sources(self):
        from src.api.diagnostic_api import deep_dive
        result = _build_result_with_observers(layers=("fc1", "fc2"))
        report = deep_dive(result)

        assert len(report.error_sources) > 0
        entry = report.error_sources[0]
        assert entry.layer != ""
        assert entry.error_source in ("Source", "Mixed", "Propagated", "Local")
        assert entry.dominant_role in ("input", "weight", "output")

    def test_auto_layer_selection(self):
        from src.api.diagnostic_api import deep_dive
        result = _build_result_with_observers(layers=("fc1", "fc2", "fc3"))
        report = deep_dive(result, k=2)

        # Should auto-select 2 worst layers
        all_layers = {d.layer for d in report.layer_diagnoses}
        assert len(all_layers) <= 2

    def test_summary_and_to_dict(self):
        from src.api.diagnostic_api import deep_dive
        result = _build_result_with_observers(layers=("fc1",))
        report = deep_dive(result, layers=["fc1"])

        text = report.summary()
        assert "Deep Dive" in text
        d = report.to_dict()
        assert "layer_diagnoses" in d


# ═══════════════════════════════════════════════════════════════════════════════
# prescribe tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrescribe:

    def test_empty_result(self):
        from src.api.diagnostic_api import prescribe
        report = prescribe(_make_result())
        assert report.boost_targets == []

    def test_boost_targets(self):
        from src.api.diagnostic_api import prescribe
        result = _build_result_with_observers(layers=("fc1", "fc2", "fc3"))
        report = prescribe(result, k=2)

        assert len(report.boost_targets) > 0
        target = report.boost_targets[0]
        assert target.layer != ""
        assert target.current_qsnr > 0
        assert target.action != ""

    def test_strategies_generated(self):
        from src.api.diagnostic_api import prescribe
        result = _build_result_with_observers(layers=("fc1", "fc2", "fc3"))
        report = prescribe(result, k=2)

        assert len(report.strategies) > 0
        strategy = report.strategies[0]
        assert strategy.strategy_type in ("mixed_precision", "transform", "format_change")
        assert len(strategy.target_layers) > 0

    def test_best_strategy(self):
        from src.api.diagnostic_api import prescribe
        result = _build_result_with_observers(layers=("fc1", "fc2", "fc3"))
        report = prescribe(result, k=2)

        if report.strategies:
            assert report.best_strategy != ""

    def test_summary_and_to_dict(self):
        from src.api.diagnostic_api import prescribe
        result = _build_result_with_observers(layers=("fc1",))
        report = prescribe(result)

        text = report.summary()
        assert "Prescription" in text
        d = report.to_dict()
        assert "boost_targets" in d


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_three_stages_compose(self):
        """coarse_pass → deep_dive → prescribe compose correctly."""
        from src.api.diagnostic_api import coarse_pass, deep_dive, prescribe

        results = {
            "W8A8": _build_result_with_observers(name="W8A8", accuracy=0.90,
                                                  fp32_accuracy=0.92, layers=("fc1", "fc2")),
            "W4A8": _build_result_with_observers(name="W4A8", accuracy=0.82,
                                                  fp32_accuracy=0.92, layers=("fc1", "fc2")),
        }

        coarse = coarse_pass(results)
        assert coarse.fp32_accuracy == 0.92
        assert len(coarse.gaps) == 2

        # Use worst config for deep dive
        worst_name = min(coarse.gaps, key=lambda g: g.avg_qsnr_db or 0).config
        worst_result = results[worst_name]

        deep = deep_dive(worst_result, k=3)
        assert len(deep.layer_diagnoses) > 0 or len(deep.error_sources) > 0

        rx = prescribe(worst_result, k=3)
        assert isinstance(rx.boost_targets, list)

    def test_three_reports_serializable(self):
        """All three report types serialize to dicts without error."""
        from src.api.diagnostic_api import coarse_pass, deep_dive, prescribe
        import json

        results = {
            "cfg": _build_result_with_observers(name="cfg", layers=("fc1",)),
        }

        coarse = coarse_pass(results)
        deep = deep_dive(results["cfg"])
        rx = prescribe(results["cfg"])

        # Should be JSON-serializable
        for report in (coarse, deep, rx):
            d = report.to_dict()
            json.dumps(d)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Extensibility tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPluggableBottleneck:

    def test_custom_bottleneck_fn(self):
        """Custom bottleneck detector overrides default."""
        from src.api.diagnostic_api import coarse_pass, BottleneckAssessment

        def custom_fn(gaps):
            return BottleneckAssessment(primary="custom", weight_degradation=0.5)

        results = {"cfg": _make_result(name="NF4", accuracy=0.80, fp32_accuracy=0.92)}
        report = coarse_pass(results, bottleneck_fn=custom_fn)
        assert report.bottleneck.primary == "custom"
        assert report.bottleneck.weight_degradation == 0.5

    def test_default_bottleneck_skips_non_wxa(self):
        """Default detector returns unknown for non-WxAy config names."""
        from src.api.diagnostic_api import coarse_pass
        results = {
            "nf4": _make_result(name="nf4", accuracy=0.80, fp32_accuracy=0.92),
        }
        report = coarse_pass(results)
        assert report.bottleneck.primary == "unknown"


class TestGracefulDegradation:

    def test_deep_dive_no_observers_still_produces_partial_report(self):
        """deep_dive with no observer data still returns error_sources and depth_decay."""
        from src.api.diagnostic_api import deep_dive
        result = _make_result(
            qsnr_per_layer={"fc1": 25.0},
            qsnr_by_role={"input": {"fc1": 20}, "weight": {"fc1": 22}, "output": {"fc1": 25}},
            accum_qsnr={"fc1": 23.0},
        )
        report = deep_dive(result)
        # No distribution diagnosis or block analysis (no observers)
        assert report.layer_diagnoses == []
        assert report.block_analyses == []
        # But error_sources should still be populated
        assert len(report.error_sources) > 0

    def test_prescribe_no_observers_still_produces_targets(self):
        """prescribe with no observer data still returns partial results."""
        from src.api.diagnostic_api import prescribe
        result = _make_result(
            qsnr_per_layer={"fc1": 25.0},
            qsnr_by_role={"input": {"fc1": 20}, "weight": {"fc1": 22}, "output": {"fc1": 25}},
        )
        report = prescribe(result)
        assert isinstance(report.boost_targets, list)
        assert isinstance(report.strategies, list)

    def test_coarse_pass_with_accuracy_only(self):
        """coarse_pass works with only accuracy data (no QSNR)."""
        from src.api.diagnostic_api import coarse_pass
        results = {
            "int4": _make_result(name="int4", accuracy=0.75, fp32_accuracy=0.92),
            "int8": _make_result(name="int8", accuracy=0.90, fp32_accuracy=0.92),
        }
        report = coarse_pass(results)
        assert len(report.gaps) == 2
        assert report.gaps[0].accuracy is not None or report.gaps[1].accuracy is not None


# ═══════════════════════════════════════════════════════════════════════════════
# save_diagnostic_data + dist_overlay tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveDiagnosticData:

    @pytest.fixture
    def full_pipeline_reports(self, tmp_path):
        """Run full pipeline and save, return (coarse, deep, rx, diagnostic_dir)."""
        from src.api.diagnostic_api import coarse_pass, deep_dive, prescribe, save_diagnostic_data

        results = {
            "W8A8": _build_result_with_observers(name="W8A8", accuracy=0.90,
                                                  fp32_accuracy=0.92, layers=("fc1", "fc2")),
            "W4A8": _build_result_with_observers(name="W4A8", accuracy=0.82,
                                                  fp32_accuracy=0.92, layers=("fc1", "fc2")),
        }

        coarse = coarse_pass(results)
        worst_name = min(coarse.gaps, key=lambda g: g.avg_qsnr_db or 0).config
        worst_result = results[worst_name]

        deep = deep_dive(worst_result, k=3)
        rx = prescribe(worst_result, k=3)

        diag_dir = save_diagnostic_data(coarse, deep, rx, str(tmp_path))
        return coarse, deep, rx, diag_dir

    def test_directory_structure(self, full_pipeline_reports):
        _, _, _, diag_dir = full_pipeline_reports
        assert os.path.isfile(f"{diag_dir}/index.json")
        assert os.path.isfile(f"{diag_dir}/coarse/gaps.json")
        assert os.path.isfile(f"{diag_dir}/coarse/bottleneck.json")
        assert os.path.isfile(f"{diag_dir}/deep_dive/index.json")
        assert os.path.isfile(f"{diag_dir}/deep_dive/depth_decay.json")
        assert os.path.isfile(f"{diag_dir}/prescription/boost_targets.json")

    def test_index_has_catalog(self, full_pipeline_reports):
        _, _, _, diag_dir = full_pipeline_reports
        with open(f"{diag_dir}/index.json") as f:
            index = json.load(f)
        assert "fp32_accuracy" in index
        assert "config_names" in index
        assert "available_data" in index
        # Verify all three sections
        for section in ("coarse", "deep_dive", "prescription"):
            assert section in index["available_data"]
            assert "description" in index["available_data"][section]

    def test_coarse_files_valid_json(self, full_pipeline_reports):
        _, _, _, diag_dir = full_pipeline_reports
        coarse_files = ["gaps", "bottleneck", "consistent_worst",
                        "config_specific_worst", "transform_effects",
                        "distribution_taxonomy", "error_by_range"]
        for name in coarse_files:
            path = f"{diag_dir}/coarse/{name}.json"
            assert os.path.isfile(path), f"Missing {path}"
            with open(path) as f:
                data = json.load(f)
            assert data is not None

    def test_deep_dive_per_layer_files(self, full_pipeline_reports):
        _, deep, _, diag_dir = full_pipeline_reports
        with open(f"{diag_dir}/deep_dive/index.json") as f:
            dd_index = json.load(f)
        assert "layers" in dd_index
        # Each layer mentioned in index should have a file
        for layer, desc in dd_index["layers"].items():
            safe = _safe_layer_name(layer)
            path = f"{diag_dir}/deep_dive/layer_{safe}.json"
            assert os.path.isfile(path), f"Missing per-layer file for {layer}"
            with open(path) as f:
                data = json.load(f)
            # Should have at least one of the expected keys
            assert any(k in data for k in ("diagnoses", "blocks", "dist_overlay"))

    def test_deep_dive_global_files(self, full_pipeline_reports):
        _, _, _, diag_dir = full_pipeline_reports
        for fname in ("depth_decay.json", "error_sources.json", "sensitivity.json"):
            path = f"{diag_dir}/deep_dive/{fname}"
            assert os.path.isfile(path)
            with open(path) as f:
                json.load(f)  # must be valid JSON

    def test_prescription_files(self, full_pipeline_reports):
        _, _, rx, diag_dir = full_pipeline_reports
        for fname in ("boost_targets.json", "strategies.json"):
            path = f"{diag_dir}/prescription/{fname}"
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)

    def test_dist_overlay_extracted(self, full_pipeline_reports):
        _, deep, _, diag_dir = full_pipeline_reports
        if not deep.dist_overlays:
            pytest.skip("No dist_overlay data (HistogramObserver not attached)")
        for layer, roles in deep.dist_overlays.items():
            safe = _safe_layer_name(layer)
            path = f"{diag_dir}/deep_dive/layer_{safe}.json"
            with open(path) as f:
                data = json.load(f)
            assert "dist_overlay" in data
            for role in roles:
                assert role in data["dist_overlay"]
                stored = data["dist_overlay"][role]
                assert "chart_data" in stored
                assert len(stored["chart_data"]) > 0
                assert "bin" in stored["chart_data"][0]

    def test_dist_overlay_to_chart_data(self):
        """DistOverlayData.to_chart_data() produces render_chart-compatible data."""
        from src.api.diagnostic_api import DistOverlayData
        od = DistOverlayData(
            bins=[-1.0, 0.0, 1.0],
            fp32=[100, 200, 100],
            quant=[95, 195, 95],
            error=[5, 5, 5],
        )
        chart_data = od.to_chart_data()
        assert len(chart_data) == 3
        assert chart_data[0]["bin"] == -1.0
        assert chart_data[0]["fp32"] == 100
        assert chart_data[0]["quant"] == 95

    def test_incremental_loading_simulation(self, full_pipeline_reports):
        """Simulate agent reading index → selecting files → getting data."""
        _, _, _, diag_dir = full_pipeline_reports

        # Step 1: Agent reads index
        with open(f"{diag_dir}/index.json") as f:
            index = json.load(f)
        assert index["bottleneck_primary"] in ("weight", "activation", "both", "unknown")

        # Step 2: Agent reads coarse gaps
        with open(f"{diag_dir}/coarse/gaps.json") as f:
            gaps = json.load(f)
        assert len(gaps) == 2

        # Step 3: Agent checks deep_dive index
        with open(f"{diag_dir}/deep_dive/index.json") as f:
            dd_index = json.load(f)
        layers = list(dd_index["layers"].keys())
        assert len(layers) > 0

        # Step 4: Agent reads one specific layer file
        safe = _safe_layer_name(layers[0])
        with open(f"{diag_dir}/deep_dive/layer_{safe}.json") as f:
            layer_data = json.load(f)
        assert isinstance(layer_data, dict)


class TestRunDiagnosticPipeline:

    def test_loads_from_results_json(self, tmp_path):
        """run_diagnostic_pipeline reads results.json and produces diagnostic/."""
        from src.api.diagnostic_api import run_diagnostic_pipeline

        # Create a minimal results.json matching StudyReport.save format
        results_data = {
            "default": {
                "W8A8": {
                    "accuracy": 0.90,
                    "fp32_accuracy": 0.92,
                    "qsnr_per_layer": {"fc1": 30.5, "fc2": 28.1},
                    "mse_per_layer": {"fc1": 0.001, "fc2": 0.003},
                },
                "W4A4": {
                    "accuracy": 0.75,
                    "fp32_accuracy": 0.92,
                    "qsnr_per_layer": {"fc1": 22.0, "fc2": 15.3},
                    "mse_per_layer": {"fc1": 0.01, "fc2": 0.05},
                },
            }
        }
        with open(f"{tmp_path}/results.json", "w") as f:
            json.dump(results_data, f)

        diag_dir = run_diagnostic_pipeline(str(tmp_path))

        assert os.path.isdir(diag_dir)
        assert os.path.isfile(f"{diag_dir}/index.json")
        assert os.path.isfile(f"{diag_dir}/coarse/gaps.json")
        assert os.path.isfile(f"{diag_dir}/deep_dive/index.json")
        assert os.path.isfile(f"{diag_dir}/prescription/strategies.json")

    def test_empty_results_raises(self, tmp_path):
        """run_diagnostic_pipeline raises on empty results."""
        from src.api.diagnostic_api import run_diagnostic_pipeline

        with open(f"{tmp_path}/results.json", "w") as f:
            json.dump({}, f)

        with pytest.raises(ValueError, match="No SessionResult"):
            run_diagnostic_pipeline(str(tmp_path))

    def test_missing_file_raises(self, tmp_path):
        """run_diagnostic_pipeline raises on missing results.json."""
        from src.api.diagnostic_api import run_diagnostic_pipeline

        with pytest.raises(FileNotFoundError):
            run_diagnostic_pipeline(str(tmp_path))
