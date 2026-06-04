"""Tests for src/api/harness_charts.py — U1–U6 + block/provenance render_chart functions."""

import math
from unittest.mock import patch, MagicMock

import pytest
import torch

from src.analysis.observers import PerBlockQSNRObserver, QSNRObserver, DistributionObserver
from src.observer.events import QuantEvent
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.scheme.quant_scheme import QuantScheme
from src.formats.base import FormatBase
from src.session import QuantConfig


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


def _make_result(observers_data=None, qsnr_per_layer=None, qsnr_by_role=None,
                 accum_qsnr=None, name="test"):
    from src.session._result import SessionResult
    config = QuantConfig(name=name, w_format="int8", w_granularity="per_block",
                         w_block_size=16, a_format="int8", a_granularity="per_block",
                         a_block_size=16)
    return SessionResult(
        name=name, config=config,
        qsnr_per_layer=qsnr_per_layer or {},
        mse_per_layer={},
        qsnr_by_role=qsnr_by_role or {},
        accum_qsnr_per_layer=accum_qsnr or {},
        observers_data=observers_data or {},
    )


def _build_obs_data_with_blocks(layer="fc1", role="weight", n_blocks=16):
    """Build observers_data with per-block QSNR entries (multiple blocks per event)."""
    obs = PerBlockQSNRObserver()
    # block_size=4, tensor size=64 → 16 blocks per event
    fp32 = torch.randn(64)
    quant = fp32 + 0.01 * torch.randn(64)
    event = _make_event(fp32, quant, layer=layer, role=role, block_size=4)
    obs.on_event(event)
    return obs.report()


def _build_obs_data_with_dist(layer="fc1", role="weight"):
    """Build observers_data with distribution metrics."""
    obs = DistributionObserver()
    fp32 = torch.randn(128)
    quant = fp32 + 0.05 * torch.randn(128)
    event = _make_event(fp32, quant, layer=layer, role=role,
                        mode=GranularityMode.PER_TENSOR)
    obs.on_event(event)
    return obs.report()


# ── Mock render_chart fixture ────────────────────────────────────────────────

@pytest.fixture
def mock_render():
    """Patch render_chart in _chart_helpers (the single source of truth)."""
    import src.api._chart_helpers as mod
    original = mod.render_chart
    m = MagicMock()
    mod.render_chart = m
    yield m
    mod.render_chart = original


@pytest.fixture
def sample_result():
    """SessionResult with per-block data (enough blocks for box chart)."""
    block_data = _build_obs_data_with_blocks("fc1", "weight", 20)

    obs_data = block_data

    return _make_result(
        observers_data=obs_data,
        qsnr_per_layer={"fc1": 25.0},
        qsnr_by_role={"input": {"fc1": 20.0}, "weight": {"fc1": 22.0}, "output": {"fc1": 25.0}},
        accum_qsnr={"fc1": 23.0},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoHarnessNoop:
    """All functions must be no-ops when render_chart is None."""

    def test_all_functions_no_crash(self, sample_result):
        """Every chart function runs without error when harness is absent."""
        from src.api.harness_charts import (
            distribution_fit_chart, intervention_chart,
            channel_heterogeneity_chart, depth_decay_chart,
            error_propagation_chart, block_qsnr_box_chart,
            block_error_chart, channel_error_chart,
            error_provenance_chart, all_harness_charts,
        )
        # render_chart is None in test environment (harness not installed)
        distribution_fit_chart(sample_result)
        intervention_chart(sample_result)
        channel_heterogeneity_chart(sample_result, "fc1")
        depth_decay_chart(sample_result)
        error_propagation_chart(sample_result)
        block_qsnr_box_chart(sample_result)
        block_error_chart(sample_result, "fc1")
        channel_error_chart(sample_result, "fc1")
        error_provenance_chart(sample_result)
        all_harness_charts(sample_result)


class TestDistributionFitChart:
    def test_no_fit_data(self, mock_render, sample_result):
        from src.api.harness_charts import distribution_fit_chart
        distribution_fit_chart(sample_result)
        mock_render.assert_not_called()

    def test_with_fit_data(self, mock_render):
        from src.api.harness_charts import distribution_fit_chart
        obs_data = {"fc1": {"weight": {"quantize": {
            ("aggregate",): {"best_fit": "norm", "best_fit_ks": 0.05, "fit_ranking": [("norm", 0.05), ("laplace", 0.12)]}
        }}}}
        result = _make_result(observers_data=obs_data)
        distribution_fit_chart(result)
        assert mock_render.call_count >= 1


class TestInterventionChart:
    def test_empty_result(self, mock_render):
        from src.api.harness_charts import intervention_chart
        result = _make_result()
        intervention_chart(result)
        # Either no call (no overrides) or table call — both OK


class TestChannelHeterogeneity:
    def test_with_block_data(self, mock_render, sample_result):
        from src.api.harness_charts import channel_heterogeneity_chart
        channel_heterogeneity_chart(sample_result, "fc1", role="weight")
        assert mock_render.call_count >= 1
        # First call should be table (stats)
        call = mock_render.call_args_list[0]
        assert call[0][1] == "table" or call[1].get("chart_type") == "table"

    def test_no_data(self, mock_render):
        from src.api.harness_charts import channel_heterogeneity_chart
        result = _make_result()
        channel_heterogeneity_chart(result, "nonexistent")
        mock_render.assert_not_called()


class TestDepthDecay:
    def test_with_data(self, mock_render, sample_result):
        from src.api.harness_charts import depth_decay_chart
        depth_decay_chart(sample_result)
        # Should emit at least line + table
        assert mock_render.call_count >= 1

    def test_empty(self, mock_render):
        from src.api.harness_charts import depth_decay_chart
        result = _make_result()
        depth_decay_chart(result)
        # No crash is sufficient


class TestErrorPropagation:
    def test_with_roles(self, mock_render, sample_result):
        from src.api.harness_charts import error_propagation_chart
        error_propagation_chart(sample_result)
        assert mock_render.call_count >= 1


class TestBlockQsnrBox:
    def test_with_blocks(self, mock_render, sample_result):
        from src.api.harness_charts import block_qsnr_box_chart
        block_qsnr_box_chart(sample_result)
        # Should emit box + table
        assert mock_render.call_count >= 1
        # Check that box chart_type was used
        chart_types = [c[0][1] for c in mock_render.call_args_list]
        assert "box" in chart_types

    def test_box_data_format(self, mock_render, sample_result):
        """Box chart data should have 'group' and 'value' columns."""
        from src.api.harness_charts import block_qsnr_box_chart
        block_qsnr_box_chart(sample_result)
        for call in mock_render.call_args_list:
            data = call[0][0]
            chart_type = call[0][1]
            if chart_type == "box":
                assert all("group" in row and "value" in row for row in data)


class TestBlockErrorChart:
    def test_with_data(self, mock_render, sample_result):
        from src.api.harness_charts import block_error_chart
        block_error_chart(sample_result, "fc1", role="weight")
        assert mock_render.call_count >= 1


class TestChannelErrorChart:
    def test_with_data(self, mock_render, sample_result):
        from src.api.harness_charts import channel_error_chart
        channel_error_chart(sample_result, "fc1", role="weight")
        assert mock_render.call_count >= 1


class TestErrorProvenanceChart:
    def test_with_roles(self, mock_render, sample_result):
        from src.api.harness_charts import error_provenance_chart
        error_provenance_chart(sample_result)
        assert mock_render.call_count >= 1


class TestAllHarnessCharts:
    def test_runs_all(self, mock_render, sample_result):
        from src.api.harness_charts import all_harness_charts
        all_harness_charts(sample_result, label="Test")
        # Pruned to U2a (intervention table) + U6 (box plot) only
        assert mock_render.call_count >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases & additional coverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoxGuard:
    def test_box_skipped_with_fewer_than_3_rows(self, mock_render):
        """_chart should suppress box when < 3 rows."""
        from src.api._chart_helpers import _chart
        _chart([{"g": "A", "v": 1}], "box", x="g", y="v")
        mock_render.assert_not_called()

    def test_box_emits_with_3_rows(self, mock_render):
        from src.api._chart_helpers import _chart
        _chart([{"g": "A", "v": 1}, {"g": "A", "v": 2}, {"g": "A", "v": 3}],
               "box", x="g", y="v")
        assert mock_render.call_count == 1


class TestEdgeCases:
    def test_empty_observers_data(self, mock_render):
        """All functions should be no-op with empty observers_data."""
        from src.api.harness_charts import (
            error_propagation_chart, error_provenance_chart, block_qsnr_box_chart,
        )
        result = _make_result()
        error_propagation_chart(result)
        error_provenance_chart(result)
        block_qsnr_box_chart(result)
        mock_render.assert_not_called()

    def test_nan_qsnr_values(self, mock_render):
        """NaN QSNR should be filtered out, not crash."""
        from src.api.harness_charts import error_propagation_chart
        result = _make_result(
            observers_data={"fc1": {"weight": {"quantize": {("aggregate",): {"qsnr_db": float("nan")}}}}},
            qsnr_by_role={"output": {"fc1": float("nan")}},
        )
        error_propagation_chart(result)
        # Should not crash; may or may not emit depending on NaN filter

    def test_single_layer_model(self, mock_render):
        """Single-layer model should work without crash."""
        from src.api.harness_charts import error_provenance_chart
        result = _make_result(
            qsnr_by_role={"input": {"fc1": 20.0}, "weight": {"fc1": 25.0}, "output": {"fc1": 30.0}},
            accum_qsnr={"fc1": 28.0},
            observers_data={"fc1": {"weight": {}}},
        )
        error_provenance_chart(result)
        assert mock_render.call_count >= 1


class TestErrorSourceClassification:
    def test_source_classification_helper(self):
        from src.api.harness_charts import _classify_error_source
        assert _classify_error_source(None, 20.0) == "Local"
        assert _classify_error_source(30.0, 28.0) == "Source"    # diff=2.0 < 3.0
        assert _classify_error_source(30.0, 22.0) == "Mixed"    # diff=8.0
        assert _classify_error_source(30.0, 15.0) == "Propagated"  # diff=15.0


class TestU5vsProvenanceDistinction:
    def test_u5_produces_table_only(self, mock_render, sample_result):
        """U5 should only produce source classification table, not per-role bar."""
        from src.api.harness_charts import error_propagation_chart
        error_propagation_chart(sample_result)
        chart_types = [c[0][1] for c in mock_render.call_args_list]
        assert "table" in chart_types
        assert "bar" not in chart_types  # U5 no longer produces bar

    def test_provenance_produces_bar(self, mock_render, sample_result):
        """Provenance should produce per-role bar + table."""
        from src.api.harness_charts import error_provenance_chart
        error_provenance_chart(sample_result)
        chart_types = [c[0][1] for c in mock_render.call_args_list]
        assert "bar" in chart_types
        assert "table" in chart_types


class TestQSNRRef:
    def test_constant_available(self):
        from src.api._chart_helpers import QSNR_REF
        assert QSNR_REF == 60.0


class TestSharedHelpers:
    def test_block_stats_with_p10_p90(self):
        from src.api._chart_helpers import _block_stats
        blocks = {i: float(i) for i in range(100)}
        stats = _block_stats(blocks)
        assert "p10" in stats
        assert "p90" in stats
        assert stats["n_blocks"] == 100

    def test_get_per_channel_qsnr(self):
        from src.api._chart_helpers import _get_per_channel_qsnr
        obs_data = {"fc1": {"weight": {"quantize": {
            ("channel", 0): {"qsnr_db": 30.0, "mse": 0.001},
            ("channel", 1): {"qsnr_db": 25.0, "mse": 0.003},
        }}}}
        result = _get_per_channel_qsnr(obs_data, "fc1", "weight")
        assert result == {0: 30.0, 1: 25.0}
