"""Tests for src/analysis/compare.py — ComparisonReport and compare_formats."""
import pytest
import torch
import torch.nn as nn

from src.analysis.compare import higher_is_better, ComparisonReport, compare_formats
from src.analysis.report import Report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_report(mse_vals, qsnr_vals):
    """Build a minimal Report with one layer per entry."""
    raw = {}
    for i, (mse, qsnr) in enumerate(zip(mse_vals, qsnr_vals)):
        raw[f"layer{i}"] = {
            "input": {
                "input_pre_quant[0]": {("tensor",): {"mse": mse, "qsnr_db": qsnr}},
            }
        }
    return Report(raw)


# ===================================================================
# higher_is_better
# ===================================================================

class TestHigherIsBetter:
    def test_known_metrics(self):
        assert higher_is_better("qsnr_db") is True
        assert higher_is_better("mse") is False
        assert higher_is_better("dynamic_range_bits") is False

    def test_unknown_metric_defaults_true(self):
        assert higher_is_better("unknown_metric") is True


# ===================================================================
# ComparisonReport
# ===================================================================

class TestComparisonReport:
    def test_to_dataframe(self):
        r1 = _make_report([0.01, 0.02], [20.0, 18.0])
        r2 = _make_report([0.005, 0.015], [25.0, 22.0])
        cr = ComparisonReport({"fmt_a": r1, "fmt_b": r2})
        df = cr.to_dataframe()
        assert len(df) > 0
        # If pandas is available, check the format column
        if hasattr(df, "columns"):
            assert "format" in df.columns
            assert set(df["format"].unique()) == {"fmt_a", "fmt_b"}

    def test_summary(self):
        r1 = _make_report([0.01, 0.02], [20.0, 18.0])
        r2 = _make_report([0.005], [30.0])
        cr = ComparisonReport({"a": r1, "b": r2})
        s = cr.summary()
        assert "a" in s and "b" in s
        assert s["a"]["total_layers"] == 2
        assert s["b"]["total_layers"] == 1

    def test_rank_formats_qsnr(self):
        r1 = _make_report([0.01], [20.0])
        r2 = _make_report([0.01], [30.0])
        cr = ComparisonReport({"low": r1, "high": r2})
        ranked = cr.rank_formats(metric="qsnr_db")
        assert ranked[0][0] == "high"

    def test_rank_formats_mse(self):
        r1 = _make_report([0.10], [10.0])
        r2 = _make_report([0.01], [10.0])
        cr = ComparisonReport({"bad": r1, "good": r2})
        ranked = cr.rank_formats(metric="mse")
        # mse: lower is better
        assert ranked[0][0] == "good"

    def test_rank_formats_by_role(self):
        raw_a = {"L0": {"input": {"s0": {("t",): {"qsnr_db": 20.0}}}}}
        raw_b = {"L0": {"input": {"s0": {("t",): {"qsnr_db": 30.0}}}}}
        cr = ComparisonReport({"a": Report(raw_a), "b": Report(raw_b)})
        ranked = cr.rank_formats(metric="qsnr_db", role="input")
        assert ranked[0][0] == "b"

    def test_recommend_per_layer(self):
        raw_a = {"L": {"input": {"s": {("t",): {"qsnr_db": 20.0}}}}}
        raw_b = {"L": {"input": {"s": {("t",): {"qsnr_db": 30.0}}}}}
        cr = ComparisonReport({"a": Report(raw_a), "b": Report(raw_b)})
        recs = cr.recommend(metric="qsnr_db")
        assert recs["L"]["best_format"] == "b"
        assert "scores_by_format" in recs["L"]
        assert recs["L"]["scores_by_format"]["a"] == pytest.approx(20.0)
        assert recs["L"]["scores_by_format"]["b"] == pytest.approx(30.0)

    def test_recommend_mse_lower_is_better(self):
        raw_a = {"L": {"input": {"s": {("t",): {"mse": 0.01}}}}}
        raw_b = {"L": {"input": {"s": {("t",): {"mse": 0.10}}}}}
        cr = ComparisonReport({"a": Report(raw_a), "b": Report(raw_b)})
        recs = cr.recommend(metric="mse")
        assert recs["L"]["best_format"] == "a"

    def test_print_comparison_no_crash(self):
        r = _make_report([0.01], [20.0])
        cr = ComparisonReport({"fmt": r})
        cr.print_comparison()

    def test_empty_reports(self):
        cr = ComparisonReport({})
        assert cr.summary() == {}
        assert cr.rank_formats() == []
        assert cr.recommend() == {}


# ===================================================================
# compare_formats
# ===================================================================

class TestCompareFormats:
    def test_requires_at_least_one_config(self):
        with pytest.raises(ValueError, match="at least one configuration"):
            compare_formats(lambda n, c: nn.Linear(1, 1), torch.randn(2, 1), {})

    def test_runs_and_returns_comparison(self):
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec
        from src.session._model import quantize_model

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                return self.fc(x)

        scheme = QuantScheme(
            format="fp8_e4m3",
            granularity=GranularitySpec.per_tensor(),
        )
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        def build(name, config):
            return quantize_model(Tiny(), config)

        data = torch.randn(3, 4)
        result = compare_formats(build, data, {"cfg_a": cfg})

        assert isinstance(result, ComparisonReport)
        assert len(result.reports) == 1
        assert "cfg_a" in result.reports

    def test_handles_list_of_batches(self):
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec
        from src.session._model import quantize_model

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2)

            def forward(self, x):
                return self.fc(x)

        scheme = QuantScheme(
            format="fp8_e4m3",
            granularity=GranularitySpec.per_tensor(),
        )
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        def build(name, config):
            return quantize_model(Tiny(), config)

        batches = [torch.randn(2, 2), torch.randn(2, 2)]
        result = compare_formats(build, batches, {"x": cfg})
        assert isinstance(result, ComparisonReport)

    def test_eval_fn_used_when_provided(self):
        from src.scheme.op_config import OpQuantConfig
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec
        from src.session._model import quantize_model

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                return self.fc(x)

        scheme = QuantScheme(
            format="fp8_e4m3",
            granularity=GranularitySpec.per_tensor(),
        )
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

        def build(name, config):
            return quantize_model(Tiny(), config)

        called = []

        def my_eval(model, batch):
            called.append(1)
            return model(batch)

        data = torch.randn(3, 4)
        result = compare_formats(build, data, {"cfg_a": cfg}, eval_fn=my_eval)
        assert len(called) > 0
