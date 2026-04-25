import torch
import torch.nn as nn
import pytest

from src.analysis.compare import compare_formats, ComparisonReport
from src.analysis.observers import QSNRObserver, DistributionObserver
from src.analysis.report import Report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(fmt_name="fp8_e4m3", mode="per_tensor", block_size=None):
    from src.scheme.quant_scheme import QuantScheme
    from src.scheme.op_config import OpQuantConfig
    from src.formats.base import FormatBase
    from src.scheme.granularity import GranularitySpec, GranularityMode
    from src.scheme.transform import IdentityTransform

    fmt = FormatBase.from_str(fmt_name)
    if mode == "per_tensor":
        gran = GranularitySpec(mode=GranularityMode.PER_TENSOR)
    elif mode == "per_channel":
        gran = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0)
    elif mode == "per_block":
        gran = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=block_size or 32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scheme = QuantScheme(format=fmt, granularity=gran, transform=IdentityTransform())
    return OpQuantConfig(input=(scheme,), weight=(scheme,))


def _build_two_layer_linear(cfg_name, config):
    """Build a fresh quantized 2-layer linear model for the given config."""
    from src.ops.linear import QuantizedLinear

    class TwoLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = QuantizedLinear(8, 4, bias=False, cfg=config, name="layer0")
            self.layer1 = QuantizedLinear(4, 2, bias=False, cfg=config, name="layer1")

        def forward(self, x):
            return self.layer1(self.layer0(x))

    return TwoLayer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompareFormats:
    def test_compare_two_formats(self):
        configs = {
            "fp8_per_tensor": _make_config("fp8_e4m3", "per_tensor"),
            "fp8_per_channel": _make_config("fp8_e4m3", "per_channel"),
        }

        x = torch.randn(3, 8)
        observers = [QSNRObserver(), DistributionObserver()]

        result = compare_formats(_build_two_layer_linear, x, configs, observers)

        assert isinstance(result, ComparisonReport)
        assert len(result.reports) == 2

        # to_dataframe
        df = result.to_dataframe()
        assert len(df) > 0
        assert "format" in df.columns
        assert "layer" in df.columns
        assert "role" in df.columns

    def test_summary_aggregates_per_format(self):
        configs = {
            "fp8": _make_config("fp8_e4m3", "per_tensor"),
            "int8": _make_config("int8", "per_tensor"),
        }

        x = torch.randn(3, 8)
        result = compare_formats(_build_two_layer_linear, x, configs)

        summary = result.summary()
        assert "fp8" in summary
        assert "int8" in summary
        for fmt_name, stats in summary.items():
            assert "avg_qsnr_db" in stats
            assert "avg_mse" in stats
            assert stats["total_layers"] > 0

    def test_rank_formats(self):
        configs = {
            "fp8": _make_config("fp8_e4m3", "per_tensor"),
            "int8": _make_config("int8", "per_tensor"),
        }

        x = torch.randn(3, 8)
        result = compare_formats(_build_two_layer_linear, x, configs)

        ranked = result.rank_formats(metric="qsnr_db")
        assert len(ranked) == 2
        assert ranked[0][0] in ("fp8", "int8")
        assert isinstance(ranked[0][1], float)

    def test_recommend_per_layer(self):
        configs = {
            "fp8": _make_config("fp8_e4m3", "per_tensor"),
            "int8": _make_config("int8", "per_tensor"),
        }

        x = torch.randn(3, 8)
        result = compare_formats(_build_two_layer_linear, x, configs)

        recs = result.recommend()
        assert len(recs) > 0
        for layer, rec in recs.items():
            assert "best_format" in rec
            assert rec["best_format"] in ("fp8", "int8")

    def test_print_does_not_crash(self):
        configs = {
            "fp8": _make_config("fp8_e4m3", "per_tensor"),
        }

        x = torch.randn(3, 8)
        result = compare_formats(_build_two_layer_linear, x, configs)
        result.print_comparison()

    def test_empty_configs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            compare_formats(_build_two_layer_linear, torch.randn(3, 8), {})

    def test_single_batch_or_list(self):
        configs = {"fp8": _make_config("fp8_e4m3", "per_tensor")}
        x = torch.randn(3, 8)

        # Single tensor gets wrapped into list internally
        result = compare_formats(_build_two_layer_linear, x, configs)
        assert len(result.reports) == 1

        # Explicit list also works
        result2 = compare_formats(_build_two_layer_linear, [x], configs)
        assert len(result2.reports) == 1
