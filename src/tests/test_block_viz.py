"""Tests for block error visualization functions."""
import pytest
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.session import Session, QuantConfig
from src.analysis.observers import PerBlockQSNRObserver
from src.api.block_error_analysis import block_error_analysis
from src.viz.block_error_heatmap import (
    block_error_heatmap,
    channel_error_bar,
    multi_config_block_comparison,
)


class TestBlockErrorHeatmap:
    def test_basic_heatmap_returns_figure(self):
        """Per-block data → returns a matplotlib Figure."""
        model = nn.Linear(32, 16).eval()
        config = QuantConfig(name="int4", w_format="int4", w_granularity="per_block",
                             w_block_size=8, a_format="int4", a_granularity="per_block",
                             a_block_size=8)
        session = Session(model, config, observers=[PerBlockQSNRObserver()], keep_fp32=True)
        result = session.run([torch.randn(2, 32)])

        fig = block_error_heatmap(result, layer="", role="weight")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_data_returns_figure(self):
        """No observer data → returns figure with message."""
        from src.session._result import SessionResult
        result = SessionResult(
            name="test",
            config=QuantConfig(name="test", w_format="int8", w_granularity="per_block",
                               w_block_size=16, a_format="int8", a_granularity="per_block",
                               a_block_size=16),
            qsnr_per_layer={}, mse_per_layer={}, observers_data={},
        )
        fig = block_error_heatmap(result, layer="fc1", role="weight")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestChannelErrorBar:
    def test_basic_bar_chart(self):
        """Returns a bar chart figure."""
        model = nn.Linear(32, 16).eval()
        config = QuantConfig(name="int4", w_format="int4", w_granularity="per_block",
                             w_block_size=8, a_format="int4", a_granularity="per_block",
                             a_block_size=8)
        session = Session(model, config, observers=[PerBlockQSNRObserver()], keep_fp32=True)
        result = session.run([torch.randn(2, 32)])

        fig = channel_error_bar(result, layer="", role="weight", top_k=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_top_k_respected(self):
        """Figure has bars with correct layer name."""
        model = nn.Linear(32, 16).eval()
        config = QuantConfig(name="int4", w_format="int4", w_granularity="per_block",
                             w_block_size=8, a_format="int4", a_granularity="per_block",
                             a_block_size=8)
        session = Session(model, config, observers=[PerBlockQSNRObserver()], keep_fp32=True)
        result = session.run([torch.randn(2, 32)])

        # Use the actual layer name from the quantized model
        layer_name = list(result.observers_data.keys())[0]
        fig = channel_error_bar(result, layer=layer_name, role="weight", top_k=5)
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)


class TestMultiConfigBlockComparison:
    def test_comparison_with_study(self):
        """Multi-config comparison returns grouped bar chart."""
        from src.session._study import Study
        from src.report._study_report import StudyReport

        model = nn.Linear(32, 16).eval()
        configs = [
            QuantConfig(name="W8A8", w_format="int8", w_granularity="per_block",
                        w_block_size=8, a_format="int8", a_granularity="per_block",
                        a_block_size=8),
            QuantConfig(name="W4A4", w_format="int4", w_granularity="per_block",
                        w_block_size=8, a_format="int4", a_granularity="per_block",
                        a_block_size=8),
        ]
        study = Study(configs, model=model)
        report = study.run([torch.randn(2, 32)])

        fig = multi_config_block_comparison(report, layer="", role="weight", top_k=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_matching_layer(self):
        """No data for the layer → returns empty figure."""
        from src.session._study import Study
        model = nn.Linear(32, 16).eval()
        configs = [
            QuantConfig(name="W4A4", w_format="int4", w_granularity="per_block",
                        w_block_size=8, a_format="int4", a_granularity="per_block",
                        a_block_size=8),
        ]
        study = Study(configs, model=model)
        report = study.run([torch.randn(2, 32)])

        fig = multi_config_block_comparison(report, layer="nonexistent", role="weight")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
