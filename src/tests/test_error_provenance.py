"""Tests for src/analysis/_error_provenance.py — ErrorProvenance accessor."""

import pytest

from src.analysis._error_provenance import (
    ErrorProvenance,
    _infer_layer_type,
    _select_dominant,
)
from src.session._config import QuantConfig
from src.session._result import SessionResult


# ---------------------------------------------------------------------------
# _infer_layer_type
# ---------------------------------------------------------------------------

class TestInferLayerType:
    def test_linear(self):
        assert _infer_layer_type("layer1.linear") == "Linear"

    def test_fc_maps_to_linear(self):
        assert _infer_layer_type("model.fc") == "Linear"

    def test_conv(self):
        assert _infer_layer_type("backbone.conv1") == "Conv"

    def test_norm_variants(self):
        assert _infer_layer_type("encoder.layernorm") == "Norm"
        assert _infer_layer_type("bn1") == "Norm"
        assert _infer_layer_type("rms_norm") == "Norm"
        assert _infer_layer_type("group_norm") == "Norm"

    def test_activation(self):
        assert _infer_layer_type("activation_1") == "Activation"
        assert _infer_layer_type("gelu_1") == "Activation"
        assert _infer_layer_type("relu") == "Activation"

    def test_softmax(self):
        assert _infer_layer_type("attn.softmax") == "Softmax"

    def test_pool(self):
        assert _infer_layer_type("pool") == "Pool"

    def test_embed(self):
        assert _infer_layer_type("token_embed") == "Embed"

    def test_unknown(self):
        assert _infer_layer_type("foobar") == "Other"


# ---------------------------------------------------------------------------
# _select_dominant
# ---------------------------------------------------------------------------

class TestSelectDominant:
    def test_picks_lowest_qsnr_role(self):
        qsnr_by_role = {
            "input": {"a": 30.0, "b": 20.0},
            "weight": {"a": 15.0, "b": 25.0},
            "output": {"a": 40.0, "b": 10.0},
        }
        dominant = _select_dominant(qsnr_by_role, ["a", "b"])
        assert dominant["a"] == "weight"   # 15.0 < 30.0, 40.0
        assert dominant["b"] == "output"   # 10.0 < 20.0, 25.0

    def test_skips_none_values(self):
        qsnr_by_role = {
            "input": {"a": 30.0},
            "weight": {"a": None},
            "output": {},
        }
        dominant = _select_dominant(qsnr_by_role, ["a"])
        assert dominant["a"] == "input"

    def test_no_data_returns_unknown(self):
        dominant = _select_dominant({}, ["x"])
        assert dominant["x"] == "?"


# ---------------------------------------------------------------------------
# Helpers to build SessionResult with qsnr_by_role
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> SessionResult:
    defaults = dict(
        name="test",
        config=QuantConfig(name="int8", w_format="int8", w_granularity="per_tensor"),
        qsnr_by_role={
            "input": {
                "layer0.linear": 25.0,
                "layer1.conv": 18.0,
                "layer2.norm": 45.0,
                "layer3.linear": 12.0,
            },
            "weight": {
                "layer0.linear": 30.0,
                "layer1.conv": 22.0,
                "layer2.norm": 50.0,
                "layer3.linear": 8.0,
            },
            "output": {
                "layer0.linear": 28.0,
                "layer1.conv": 20.0,
                "layer2.norm": 48.0,
                "layer3.linear": 10.0,
            },
        },
    )
    defaults.update(overrides)
    return SessionResult(**defaults)


# ---------------------------------------------------------------------------
# ErrorProvenance — summary
# ---------------------------------------------------------------------------

class TestErrorProvenanceSummary:
    def test_returns_formatted_table(self):
        result = _make_result()
        prov = ErrorProvenance(result)
        s = prov.summary()
        assert isinstance(s, str)
        assert "Role" in s
        assert "Avg QSNR" in s
        assert "Linear" in s
        assert "Conv" in s
        assert "Norm" in s

    def test_no_qsnr_data(self):
        result = _make_result(qsnr_by_role={})
        prov = ErrorProvenance(result)
        s = prov.summary()
        assert "No per-role QSNR" in s

    def test_all_inf_or_nan_data(self):
        result = _make_result(qsnr_by_role={
            "input": {"layer0.linear": float("inf")},
            "weight": {"layer0.linear": float("nan")},
            "output": {},
        })
        prov = ErrorProvenance(result)
        s = prov.summary()
        assert "No finite QSNR" in s

    def test_single_value_std_zero(self):
        result = _make_result(qsnr_by_role={
            "input": {"layer0.linear": 25.0},
            "weight": {},
            "output": {},
        })
        prov = ErrorProvenance(result)
        s = prov.summary()
        assert "25.0" in s
        assert "0.0" in s  # std of single value


# ---------------------------------------------------------------------------
# ErrorProvenance — per_role_table
# ---------------------------------------------------------------------------

class TestErrorProvenancePerRoleTable:
    def test_returns_formatted_table(self):
        result = _make_result()
        prov = ErrorProvenance(result)
        t = prov.per_role_table()
        assert isinstance(t, str)
        assert "Layer" in t
        assert "Input" in t
        assert "Weight" in t
        assert "Output" in t
        assert "Dominant" in t

    def test_sorted_by_worst_first(self):
        """Worst-QSNR layers should appear first."""
        result = _make_result()
        prov = ErrorProvenance(result)
        t = prov.per_role_table()
        lines = t.split("\n")
        # layer3.linear has worst QSNR (8.0 weight), should be first data line
        data_lines = [l for l in lines if l.startswith("layer")]
        assert "layer3.linear" in data_lines[0]

    def test_no_qsnr_data(self):
        result = _make_result(qsnr_by_role={})
        prov = ErrorProvenance(result)
        t = prov.per_role_table()
        assert "No per-role QSNR" in t

    def test_max_layers_truncation(self):
        result = _make_result()
        prov = ErrorProvenance(result)
        t = prov.per_role_table(max_layers=2)
        data_lines = [l for l in t.split("\n") if l.startswith("layer")]
        assert len(data_lines) == 2
        assert "more layers" in t

    def test_none_values_shown_as_na(self):
        result = _make_result(qsnr_by_role={
            "input": {"layer0.linear": 25.0},
            "weight": {},
            "output": {"layer0.linear": None},
        })
        prov = ErrorProvenance(result)
        t = prov.per_role_table()
        assert "N/A" in t


# ---------------------------------------------------------------------------
# ErrorProvenance — top_k
# ---------------------------------------------------------------------------

class TestErrorProvenanceTopK:
    def test_returns_k_worst_for_role(self):
        result = _make_result()
        prov = ErrorProvenance(result)
        top = prov.top_k(k=2, role="weight")
        assert len(top) == 2
        assert top[0][0] == "layer3.linear"  # 8.0 dB worst
        assert top[1][0] == "layer1.conv"    # 22.0 dB

    def test_auto_role_picks_worst_per_layer(self):
        result = _make_result()
        prov = ErrorProvenance(result)
        top = prov.top_k(k=3, role="auto")
        # layer3.linear: worst=8.0 (weight), layer1.conv: worst=18.0 (input),
        # layer0.linear: worst=25.0 (input)
        assert len(top) == 3
        assert top[0] == ("layer3.linear", 8.0)
        assert top[1] == ("layer1.conv", 18.0)
        assert top[2] == ("layer0.linear", 25.0)

    def test_empty_qsnr_returns_empty_list(self):
        result = _make_result(qsnr_by_role={})
        prov = ErrorProvenance(result)
        assert prov.top_k(5, role="weight") == []

    def test_empty_role_returns_empty_list(self):
        result = _make_result(qsnr_by_role={"input": {"a": 1.0}})
        prov = ErrorProvenance(result)
        assert prov.top_k(5, role="weight") == []

    def test_auto_role_no_data(self):
        result = _make_result(qsnr_by_role={})
        prov = ErrorProvenance(result)
        assert prov.top_k(5, role="auto") == []

    def test_k_larger_than_data(self):
        result = _make_result()
        prov = ErrorProvenance(result)
        top = prov.top_k(k=100, role="weight")
        assert len(top) == 4  # only 4 layers in data

    def test_nan_values_filtered(self):
        result = _make_result(qsnr_by_role={
            "input": {"a": float("nan"), "b": 20.0},
            "weight": {"a": 15.0, "b": float("nan")},
            "output": {},
        })
        prov = ErrorProvenance(result)
        top = prov.top_k(k=5, role="input")
        assert len(top) == 1
        assert top[0] == ("b", 20.0)


# ---------------------------------------------------------------------------
# ErrorProvenance — depth_decay_data
# ---------------------------------------------------------------------------

class TestErrorProvenanceDepthDecayData:
    def test_returns_indexed_tuples(self):
        result = _make_result(qsnr_by_role={
            "output": {
                "layer0.linear": 28.0,
                "layer1.conv": 20.0,
                "layer2.norm": 48.0,
            },
        })
        prov = ErrorProvenance(result)
        data = prov.depth_decay_data(role="output")
        assert len(data) == 3
        assert data[0] == (0, "layer0.linear", 28.0)
        assert data[1] == (1, "layer1.conv", 20.0)
        assert data[2] == (2, "layer2.norm", 48.0)

    def test_falls_back_to_accum_qsnr(self):
        result = _make_result(
            qsnr_by_role={},
            accum_qsnr_per_layer={
                "layer0.linear": 18.0,
                "layer1.conv": 12.0,
            },
        )
        prov = ErrorProvenance(result)
        data = prov.depth_decay_data(role="output")
        assert len(data) == 2
        assert data[0] == (0, "layer0.linear", 18.0)

    def test_empty_returns_empty_list(self):
        result = _make_result(qsnr_by_role={}, accum_qsnr_per_layer={})
        prov = ErrorProvenance(result)
        assert prov.depth_decay_data() == []

    def test_filters_inf_and_nan(self):
        result = _make_result(qsnr_by_role={
            "output": {
                "a": float("inf"),
                "b": float("-inf"),
                "c": float("nan"),
                "d": 25.0,
            },
        })
        prov = ErrorProvenance(result)
        data = prov.depth_decay_data(role="output")
        assert len(data) == 1
        assert data[0][1] == "d"
