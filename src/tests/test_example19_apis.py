"""Tests for Example 19 APIs: PerBlockQSNRObserver, block_error_analysis,
CrossConfigLayerRanking, TransformEffectReport."""
import math

import pytest
import torch
import torch.nn as nn

from src.analysis.observers import PerBlockQSNRObserver, QSNRObserver
from src.api.block_error_analysis import block_error_analysis, BlockErrorReport
from src.analysis.cross_config_ranking import CrossConfigLayerRanking
from src.analysis.transform_effect import TransformEffectReport
from src.observer.events import QuantEvent
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.formats.base import FormatBase
from src.session import Session, QuantConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_event(fp32, quant, layer="test_layer", role="weight",
                mode=GranularityMode.PER_BLOCK, block_size=16, pipeline_index=0):
    """Create a minimal QuantEvent for testing observers."""
    if mode == GranularityMode.PER_CHANNEL:
        gran = GranularitySpec(mode=mode, channel_axis=0)
    elif mode == GranularityMode.PER_BLOCK:
        gran = GranularitySpec(mode=mode, block_size=block_size, block_axis=-1)
    elif mode == GranularityMode.PER_TENSOR:
        gran = GranularitySpec(mode=mode)
    else:
        gran = GranularitySpec(mode=mode, block_size=block_size, block_axis=-1)

    scheme = QuantScheme(
        format=FormatBase.from_str("int8"),
        granularity=gran,
    )
    return QuantEvent(
        layer_name=layer,
        role=role,
        stage="quantize",
        pipeline_index=pipeline_index,
        fp32_tensor=fp32,
        quant_tensor=quant,
        scheme=scheme,
    )


def _fake_session_result(qsnr_per_layer, name="test", observers_data=None):
    """Create a minimal SessionResult for testing."""
    from src.session._result import SessionResult
    config = QuantConfig(name=name, w_format="int8", w_granularity="per_block",
                         w_block_size=16, a_format="int8", a_granularity="per_block",
                         a_block_size=16)
    return SessionResult(
        name=name,
        config=config,
        qsnr_per_layer=qsnr_per_layer,
        mse_per_layer={},
        observers_data=observers_data or {},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PerBlockQSNRObserver
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerBlockQSNRObserver:

    def test_per_tensor_mode(self):
        """PER_TENSOR mode returns single measurement (delegates to base)."""
        obs = PerBlockQSNRObserver()
        fp32 = torch.randn(32)
        quant = fp32 + 0.01 * torch.randn(32)
        event = _make_event(fp32, quant, mode=GranularityMode.PER_TENSOR)
        obs.on_event(event)

        report = obs.report()
        assert "test_layer" in report
        key = list(list(list(report["test_layer"]["weight"].values())[0].keys())[0])
        assert key[0] == "tensor"

    def test_per_block_stores_individual_blocks(self):
        """PER_BLOCK mode stores each block separately, not aggregate."""
        obs = PerBlockQSNRObserver()
        # Shape [2, 32] → block_axis=-1, block_size=8 → 4 blocks
        fp32 = torch.randn(2, 32)
        quant = fp32 + 0.01 * torch.randn(2, 32)
        event = _make_event(fp32, quant, mode=GranularityMode.PER_BLOCK, block_size=8)
        obs.on_event(event)

        report = obs.report()
        slices = list(report["test_layer"]["weight"].values())[0]
        block_keys = [k for k in slices if k[0] == "block"]
        # 2 * 4 = 8 blocks
        assert len(block_keys) == 8
        # Each block has qsnr_db and mse
        for key in block_keys:
            assert "qsnr_db" in slices[key]
            assert "mse" in slices[key]

    def test_per_channel_stores_individual_channels(self):
        """PER_CHANNEL mode stores each channel separately."""
        obs = PerBlockQSNRObserver()
        fp32 = torch.randn(4, 16)
        quant = fp32 + 0.01 * torch.randn(4, 16)
        event = _make_event(fp32, quant, mode=GranularityMode.PER_CHANNEL)
        obs.on_event(event)

        report = obs.report()
        slices = list(report["test_layer"]["weight"].values())[0]
        ch_keys = [k for k in slices if k[0] == "channel"]
        assert len(ch_keys) == 4

    def test_per_block_partial_last_block(self):
        """PER_BLOCK with dim not divisible by block_size handles padding."""
        obs = PerBlockQSNRObserver()
        # 10 elements, block_size=4 → 3 blocks: [4], [4], [2+padded]
        fp32 = torch.randn(10)
        quant = fp32 + 0.01 * torch.randn(10)
        event = _make_event(fp32, quant, mode=GranularityMode.PER_BLOCK, block_size=4)
        obs.on_event(event)

        report = obs.report()
        slices = list(report["test_layer"]["weight"].values())[0]
        block_keys = sorted([k for k in slices if k[0] == "block"], key=lambda k: k[1])
        # 10/4 = 3 blocks (last is partial)
        assert len(block_keys) == 3
        # All QSNR values should be finite
        for key in block_keys:
            assert math.isfinite(slices[key]["qsnr_db"])

    def test_perfect_quantization_high_qsnr(self):
        """fp32 == quant → each block should have very high QSNR."""
        obs = PerBlockQSNRObserver()
        fp32 = torch.randn(2, 16)
        quant = fp32.clone()
        event = _make_event(fp32, quant, mode=GranularityMode.PER_BLOCK, block_size=8)
        obs.on_event(event)

        report = obs.report()
        slices = list(report["test_layer"]["weight"].values())[0]
        for key, metrics in slices.items():
            if key[0] == "block":
                assert metrics["qsnr_db"] > 100

    def test_known_error_per_block(self):
        """Manual QSNR check for a small block."""
        obs = PerBlockQSNRObserver()
        # Single block of 3 elements, 10% error
        fp32 = torch.tensor([1.0, 2.0, 3.0])
        quant = torch.tensor([0.9, 1.8, 2.7])
        metrics = obs._measure(("block", 0), fp32, quant)
        assert metrics["qsnr_db"] == pytest.approx(20.0, abs=0.01)

    def test_report_empty_initially(self):
        obs = PerBlockQSNRObserver()
        assert obs.report() == {}


# ═══════════════════════════════════════════════════════════════════════════════
# block_error_analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockErrorAnalysis:

    def test_basic_analysis(self):
        """Extract per-block data from a SessionResult with PerBlockQSNRObserver."""
        obs = PerBlockQSNRObserver()
        fp32 = torch.randn(2, 32)
        quant = fp32 + 0.1 * torch.randn(2, 32)
        event = _make_event(fp32, quant, layer="fc1", role="weight",
                            mode=GranularityMode.PER_BLOCK, block_size=8)
        obs.on_event(event)

        result = _fake_session_result(
            qsnr_per_layer={"fc1": 25.0},
            observers_data=obs.report(),
        )

        report = block_error_analysis(result, layer="fc1", role="weight")
        assert isinstance(report, BlockErrorReport)
        assert report.layer == "fc1"
        assert report.role == "weight"
        assert report.unit_type == "block"
        assert len(report.per_unit_qsnr) == 8  # 2 * 4 blocks
        assert len(report.worst_units) <= 10
        assert "mean" in report.stats
        assert report.stats["min"] <= report.stats["max"]

    def test_worst_units_sorted(self):
        """worst_units are sorted by QSNR ascending (worst first)."""
        obs = PerBlockQSNRObserver()
        fp32 = torch.randn(32)
        quant = fp32 + 0.1 * torch.randn(32)
        event = _make_event(fp32, quant, layer="fc1", role="weight",
                            mode=GranularityMode.PER_BLOCK, block_size=8)
        obs.on_event(event)

        result = _fake_session_result(
            qsnr_per_layer={"fc1": 20.0},
            observers_data=obs.report(),
        )

        report = block_error_analysis(result, layer="fc1", role="weight")
        qsnrs = [q for _, q in report.worst_units]
        assert qsnrs == sorted(qsnrs)

    def test_empty_observers_data(self):
        """No observer data → empty report."""
        result = _fake_session_result(qsnr_per_layer={}, observers_data={})
        report = block_error_analysis(result, layer="fc1", role="weight")
        assert report.per_unit_qsnr == {}
        assert report.worst_units == []

    def test_summary_string(self):
        obs = PerBlockQSNRObserver()
        fp32 = torch.randn(16)
        quant = fp32 + 0.1 * torch.randn(16)
        event = _make_event(fp32, quant, layer="fc1", role="weight",
                            mode=GranularityMode.PER_BLOCK, block_size=4)
        obs.on_event(event)

        result = _fake_session_result(
            qsnr_per_layer={"fc1": 20.0},
            observers_data=obs.report(),
        )
        report = block_error_analysis(result, layer="fc1", role="weight")
        s = report.summary()
        assert "fc1" in s
        assert "weight" in s


# ═══════════════════════════════════════════════════════════════════════════════
# CrossConfigLayerRanking
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossConfigLayerRanking:

    def test_consistent_worst(self):
        ranking = CrossConfigLayerRanking({
            "W8A8": {"fc1": 30.0, "fc2": 15.0, "fc3": 35.0},
            "W4A8": {"fc1": 22.0, "fc2": 10.0, "fc3": 28.0},
            "W4A4": {"fc1": 12.0, "fc2": 5.0, "fc3": 20.0},
        })
        worst = ranking.consistent_worst(k=2)
        # fc2 is worst in all configs → should appear
        layer_names = [name for name, _ in worst]
        assert "fc2" in layer_names

    def test_config_specific_worst(self):
        ranking = CrossConfigLayerRanking({
            "W8A8": {"fc1": 30.0, "fc2": 35.0, "fc3": 40.0, "fc4": 38.0},
            "W4A4": {"fc1": 30.0, "fc2": 35.0, "fc3": 40.0, "fc4": 5.0},
        })
        specific = ranking.config_specific_worst("W4A4", k=1)
        specific_names = [name for name, _ in specific]
        assert "fc4" in specific_names

    def test_layer_qsnr_delta(self):
        ranking = CrossConfigLayerRanking({
            "W8A8": {"fc1": 30.0},
            "W4A4": {"fc1": 10.0},
        })
        delta = ranking.layer_qsnr_delta("fc1", from_config="W4A4", to_config="W8A8")
        assert delta == 20.0

    def test_layer_qsnr_delta_missing(self):
        ranking = CrossConfigLayerRanking({"W8A8": {"fc1": 30.0}})
        delta = ranking.layer_qsnr_delta("fc1", from_config="W4A4", to_config="W8A8")
        assert delta is None

    def test_from_results(self):
        r1 = _fake_session_result({"fc1": 30.0, "fc2": 15.0}, name="W8A8")
        r2 = _fake_session_result({"fc1": 10.0, "fc2": 5.0}, name="W4A4")
        ranking = CrossConfigLayerRanking.from_results({"W8A8": r1, "W4A4": r2})
        assert "W8A8" in ranking.config_names
        worst = ranking.consistent_worst(k=2)
        assert len(worst) > 0

    def test_summary_string(self):
        ranking = CrossConfigLayerRanking({
            "W8A8": {"fc1": 30.0, "fc2": 15.0},
            "W4A4": {"fc1": 10.0, "fc2": 5.0},
        })
        s = ranking.summary(k=2)
        assert "W8A8" in s
        assert "W4A4" in s

    def test_empty_data(self):
        ranking = CrossConfigLayerRanking({})
        assert ranking.consistent_worst() == []


# ═══════════════════════════════════════════════════════════════════════════════
# TransformEffectReport
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformEffectReport:

    def test_auto_detect_pairs(self):
        """Config names with +SQ / +HD are auto-detected as transform variants."""
        from src.report._study_report import StudyReport

        r_base = _fake_session_result({}, name="W4A4")
        r_base.quant_metrics = {"accuracy": 0.85}
        r_base.fp32_metrics = {"accuracy": 0.95}

        r_sq = _fake_session_result({}, name="W4A4+SQ")
        r_sq.quant_metrics = {"accuracy": 0.89}
        r_sq.fp32_metrics = {"accuracy": 0.95}

        study = StudyReport({"part_0": [r_base, r_sq]})
        report = TransformEffectReport.from_study(study)

        assert len(report.pairs) == 1
        pair = report.pairs[0]
        assert pair["base_config"] == "W4A4"
        assert pair["transform"] == "smoothquant"
        assert pair["accuracy_gain"] == pytest.approx(0.04, abs=0.01)
        # Recovery: 0.04 / (0.95 - 0.85) * 100 = 40%
        assert pair["recovery_pct"] == pytest.approx(40.0, abs=1.0)

    def test_hadamard_detection(self):
        from src.report._study_report import StudyReport

        r_base = _fake_session_result({}, name="W4A4")
        r_base.quant_metrics = {"accuracy": 0.85}

        r_hd = _fake_session_result({}, name="W4A4+HD")
        r_hd.quant_metrics = {"accuracy": 0.87}

        study = StudyReport({"part_0": [r_base, r_hd]})
        report = TransformEffectReport.from_study(study)

        assert len(report.pairs) == 1
        assert report.pairs[0]["transform"] == "hadamard"
        assert report.pairs[0]["accuracy_gain"] == pytest.approx(0.02, abs=0.01)

    def test_no_pairs(self):
        """No transform configs → empty pairs."""
        from src.report._study_report import StudyReport

        r = _fake_session_result({}, name="W4A4")
        study = StudyReport({"part_0": [r]})
        report = TransformEffectReport.from_study(study)
        assert len(report.pairs) == 0

    def test_per_config_recovery(self):
        from src.report._study_report import StudyReport

        r_base = _fake_session_result({}, name="W8A8")
        r_base.quant_metrics = {"accuracy": 0.90}

        r_sq = _fake_session_result({}, name="W8A8+SQ")
        r_sq.quant_metrics = {"accuracy": 0.92}

        study = StudyReport({"part_0": [r_base, r_sq]})
        report = TransformEffectReport.from_study(study)
        recovery = report.per_config_recovery()
        assert len(recovery) == 1
        assert recovery[0]["base_config"] == "W8A8"
        assert recovery[0]["accuracy_gain"] == pytest.approx(0.02, abs=0.01)

    def test_summary_string(self):
        from src.report._study_report import StudyReport

        r_base = _fake_session_result({}, name="W4A4")
        r_base.quant_metrics = {"accuracy": 0.85}
        r_base.fp32_metrics = {"accuracy": 0.95}
        r_base.qsnr_per_layer = {"fc1": 15.0}

        r_sq = _fake_session_result({}, name="W4A4+SQ")
        r_sq.quant_metrics = {"accuracy": 0.89}
        r_sq.fp32_metrics = {"accuracy": 0.95}
        r_sq.qsnr_per_layer = {"fc1": 20.0}

        study = StudyReport({"part_0": [r_base, r_sq]})
        report = TransformEffectReport.from_study(study)
        s = report.summary()
        assert "W4A4" in s
        assert "smoothquant" in s
