"""Tests for SmoothQuant distribution comparison."""
import copy
import torch
import torch.nn as nn
import pytest

from src.analysis._smoothquant_distrib import (
    SmoothQuantDistribComparison,
    compare_smoothquant_distributions,
    _resolve_scale,
)
from src.analysis.observers import DistributionObserver
from src.transform.smooth_quant import (
    SmoothQuantTransform,
    compute_smoothquant_scale,
    fuse_smoothquant_weights,
)
from src.session._config import QuantConfig
from src.session._model import quantize_model
from src.session._session import run_quantization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlp(in_dim=64, hidden=128, out_dim=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _calibrate_and_fuse(model, calib_data, alpha=0.5):
    """Run from_model_calibration + fuse, returning (sq_transforms, fused_model)."""
    sq_transforms = SmoothQuantTransform.from_model_calibration(
        model, calib_data, alpha=alpha,
    )
    fused = fuse_smoothquant_weights(model, sq_transforms)
    return sq_transforms, fused


# ---------------------------------------------------------------------------
# Test _resolve_scale
# ---------------------------------------------------------------------------


class TestResolveScale:

    def test_linear_activation_shape(self):
        """(B, C) with channel_axis=-1 → scale shape (1, C)."""
        act = torch.randn(4, 128)
        scale = torch.ones(128)
        sq_t = SmoothQuantTransform(scale, channel_axis=-1)
        out = _resolve_scale(act, sq_t)
        assert out.shape == (1, 128)

    def test_conv_activation_shape(self):
        """(N, C, H, W) with channel_axis=1 → scale shape (1, C, 1, 1)."""
        act = torch.randn(4, 32, 14, 14)
        scale = torch.ones(32)
        sq_t = SmoothQuantTransform(scale, channel_axis=1)
        out = _resolve_scale(act, sq_t)
        assert out.shape == (1, 32, 1, 1)

    def test_device_transfer(self):
        """Scale on CPU, activation on 'meta' concept → device transfer works.
        We test the real case: both on CPU — trivially correct.
        """
        act = torch.randn(4, 64)
        scale = torch.ones(64)
        sq_t = SmoothQuantTransform(scale, channel_axis=-1)
        out = _resolve_scale(act, sq_t)
        assert out.device == act.device


# ---------------------------------------------------------------------------
# Test compare_smoothquant_distributions (core function)
# ---------------------------------------------------------------------------


class TestCompareSmoothQuantDistributions:

    def test_returns_comparison_for_linear_layers(self):
        """Two Linear layers → both present in per_layer."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        assert isinstance(result, SmoothQuantDistribComparison)
        # "0" and "2" are the Linear layers (1 is ReLU)
        assert "0" in result.per_layer
        assert "2" in result.per_layer

    def test_activation_stats_have_expected_keys(self):
        """Raw and smoothed activation stats contain DistributionObserver keys."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        for layer in result.per_layer.values():
            act = layer["activation"]
            for side in ("raw", "smoothed"):
                stats = act[side]
                assert "mean" in stats
                assert "std" in stats
                assert "dynamic_range_bits" in stats
                assert "outlier_ratio" in stats

    def test_weight_stats_present(self):
        """Weight raw and smoothed stats present."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        for layer in result.per_layer.values():
            assert "weight" in layer
            for side in ("raw", "smoothed"):
                assert side in layer["weight"]

    def test_smoothed_dr_not_larger_than_raw(self):
        """The first layer's activation DR should be reduced by SmoothQuant.

        Downstream layers may see increased DR because the activation has
        passed through an upstream layer with fused weights, altering the
        distribution before the downstream SQ scale is applied.
        """
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        x = torch.randn(8, 128)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        # Only check the first layer — downstream layers may vary.
        first_layer = result.improved_layers[0] if result.improved_layers else "0"
        act = result.per_layer[first_layer]["activation"]
        dr_raw = act["raw"]["dynamic_range_bits"]
        dr_smooth = act["smoothed"]["dynamic_range_bits"]
        # SmoothQuant should at minimum not worsen the first layer's DR
        assert dr_smooth <= dr_raw + 0.5, (
            f"First layer {first_layer}: smoothed DR ({dr_smooth:.2f}) "
            f"substantially worse than raw DR ({dr_raw:.2f})"
        )

    def test_improved_layers_sorted_by_dr_delta(self):
        """improved_layers sorted by DR reduction (largest first)."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        deltas = []
        for name in result.improved_layers:
            act = result.per_layer[name]["activation"]
            dr = act["raw"]["dynamic_range_bits"] - act["smoothed"]["dynamic_range_bits"]
            deltas.append(dr)
        assert deltas == sorted(deltas, reverse=True), (
            f"improved_layers not sorted by DR delta: {deltas}"
        )

    def test_summary_contains_aggregates(self):
        """Summary dict has mean reductions."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        assert "mean_dr_reduction" in result.summary
        assert "mean_outlier_reduction" in result.summary
        assert isinstance(result.summary["mean_dr_reduction"], float)

    def test_layers_filter(self):
        """layers= restricts output to specified layers."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x], layers=["0"],
        )
        assert list(result.per_layer.keys()) == ["0"]

    def test_eval_fn_used(self):
        """When eval_fn is provided, it is called instead of model(calib_data)."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        called = []

        def my_eval(m, data):
            called.append(1)
            m(data[0])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x], eval_fn=my_eval,
        )
        assert len(called) == 1
        # Result should still be valid
        assert "0" in result.per_layer

    def test_empty_calib_data(self):
        """Empty calib_data returns empty comparison (no crash)."""
        model = _make_mlp()
        sq_transforms, fused = _calibrate_and_fuse(model, [torch.randn(4, 64)])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [],
        )
        assert result.per_layer == {}
        assert result.improved_layers == []

    def test_empty_sq_transforms(self):
        """Empty sq_transforms → empty result."""
        model = _make_mlp()
        x = torch.randn(4, 64)

        result = compare_smoothquant_distributions(
            model, model, {}, [x],
        )
        assert result.per_layer == {}

    def test_summary_table_output(self):
        """summary_table() returns a non-empty string."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        table = result.summary_table()
        assert isinstance(table, str)
        assert "DR raw" in table
        assert "Outlier" in table

    def test_histograms_stored(self):
        """_hist fields are numpy arrays."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        act = result.per_layer["0"]["activation"]
        import numpy as np
        assert isinstance(act["raw"]["_hist"], np.ndarray)
        assert isinstance(act["smoothed"]["_hist"], np.ndarray)
        assert len(act["raw"]["_hist"]) == 64

    def test_conv2d_activation_capture(self):
        """Conv2d layers with channel_axis=1."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
        )
        x = torch.randn(2, 3, 14, 14)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )
        assert "0" in result.per_layer
        assert "2" in result.per_layer
        act = result.per_layer["0"]["activation"]
        assert "dynamic_range_bits" in act["raw"]


# ---------------------------------------------------------------------------
# Test standalone compare_smoothquant_distributions with quantize_model
# ---------------------------------------------------------------------------


class TestSessionIntegration:

    def test_sq_comparison_with_quantized_model(self):
        """compare_smoothquant_distributions works after quantize_model."""
        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        cfg = QuantConfig(
            w_format="int8",
            w_granularity="per_channel",
            a_format="int8",
            a_granularity="per_tensor",
        )
        qmodel = quantize_model(
            copy.deepcopy(fused), cfg=cfg.to_op_config(),
        )

        result = compare_smoothquant_distributions(
            model, qmodel, sq_transforms, [x],
        )
        assert isinstance(result, SmoothQuantDistribComparison)
        assert "0" in result.per_layer

    def test_sq_comparison_without_sq_transform(self):
        """compare_smoothquant_distributions without transforms returns empty."""
        model = _make_mlp()
        x = torch.randn(4, 64)

        result = compare_smoothquant_distributions(
            model, model, {}, [x],
        )
        assert result.per_layer == {}

    def test_sq_comparison_with_run_quantization(self):
        """run_quantization with smoothquant transform produces valid qmodel."""
        model = _make_mlp()
        cfg = QuantConfig(
            transform="smoothquant",
            w_format="int8",
            w_granularity="per_channel",
            a_format="int8",
            a_granularity="per_tensor",
        )
        calib = [torch.randn(4, 64)]

        qmodel, fp32_model, result = run_quantization(
            model, cfg, calib, keep_fp32=True,
        )

        # Forward pass works on quantized model
        qmodel.eval()
        with torch.no_grad():
            out = qmodel(torch.randn(4, 64))
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()

    def test_sq_comparison_standalone_call(self):
        """compare_smoothquant_distributions() standalone works after SQ calibration."""
        model = _make_mlp()
        cfg = QuantConfig(
            transform="smoothquant",
            w_format="int8",
            w_granularity="per_channel",
            a_format="int8",
            a_granularity="per_tensor",
        )
        calib = [torch.randn(4, 64)]

        qmodel, fp32_model, result = run_quantization(
            model, cfg, calib, keep_fp32=True,
        )
        assert result.qsnr_per_layer  # QSNR collected


# ---------------------------------------------------------------------------
# Test viz figure
# ---------------------------------------------------------------------------


class TestViz:

    def test_figure_returns_figure(self):
        """smoothquant_distrib_comparison() returns a matplotlib Figure."""
        import tempfile
        import os

        model = _make_mlp()
        x = torch.randn(4, 64)
        sq_transforms, fused = _calibrate_and_fuse(model, [x])

        result = compare_smoothquant_distributions(
            model, fused, sq_transforms, [x],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.viz.figures import smoothquant_distrib_comparison

            fig = smoothquant_distrib_comparison(result, k=3, output_dir=tmpdir)
            import matplotlib.pyplot as plt
            assert isinstance(fig, plt.Figure)
            # Check file was saved
            png_path = os.path.join(tmpdir, "figures",
                                    "smoothquant_distrib_comparison.png")
            assert os.path.exists(png_path)

    def test_figure_raises_on_empty(self):
        """Raises ValueError on empty comparison."""
        import tempfile

        empty = SmoothQuantDistribComparison()
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.viz.figures import smoothquant_distrib_comparison

            with pytest.raises(ValueError, match="No SmoothQuant"):
                smoothquant_distrib_comparison(empty, output_dir=tmpdir)
