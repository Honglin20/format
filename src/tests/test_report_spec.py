"""Tests for src.report._spec — output specification and resolution."""

import pytest

from src.report._spec import (
    _OUTPUT_SPEC,
    PRESETS,
    resolve_outputs,
)


class TestResolveOutputs:
    """Test the resolve_outputs function."""

    def test_default_preset(self):
        """resolve_outputs("default") returns keys from default preset."""
        obs, needs_eval, needs_cost = resolve_outputs("default")
        # default = ["accuracy", "qsnr"] → qsnr observer, needs_eval from accuracy
        assert obs == {"qsnr"}
        assert needs_eval is True
        assert needs_cost is False

    def test_all_preset(self):
        """resolve_outputs("all") returns all 17 output keys."""
        obs, needs_eval, needs_cost = resolve_outputs("all")
        assert len(PRESETS["all"]) == len(_OUTPUT_SPEC)
        # All 17 unique keys
        assert set(PRESETS["all"]) == set(_OUTPUT_SPEC.keys())

    def test_accuracy_no_observers(self):
        """Accuracy output: no observers, needs_eval=True."""
        obs, needs_eval, needs_cost = resolve_outputs(["accuracy"])
        assert obs == set()
        assert needs_eval is True
        assert needs_cost is False

    def test_qsnr_observer(self):
        """QSNR output: qsnr observer, needs_eval=False."""
        obs, needs_eval, needs_cost = resolve_outputs(["qsnr"])
        assert obs == {"qsnr"}
        assert needs_eval is False
        assert needs_cost is False

    def test_histogram_qsnr_combined(self):
        """Histogram + qsnr outputs combine observers correctly."""
        obs, needs_eval, needs_cost = resolve_outputs(["histogram", "qsnr"])
        assert obs == {"histogram", "qsnr"}
        assert needs_eval is False
        assert needs_cost is False

    def test_cost_output(self):
        """Cost output: needs_cost=True, no observers."""
        obs, needs_eval, needs_cost = resolve_outputs(["cost"])
        assert obs == set()
        assert needs_eval is False
        assert needs_cost is True

    def test_accuracy_and_cost(self):
        """Accuracy + cost: needs_eval AND needs_cost."""
        obs, needs_eval, needs_cost = resolve_outputs(["accuracy", "cost"])
        assert obs == set()
        assert needs_eval is True
        assert needs_cost is True

    def test_error_dist_includes_distribution_and_mse(self):
        """Error distribution output needs distribution + mse observers."""
        obs, needs_eval, needs_cost = resolve_outputs(["error_dist"])
        assert "distribution" in obs
        assert "mse" in obs
        assert needs_eval is False
        assert needs_cost is False

    def test_sensitivity_output(self):
        """Sensitivity output: needs qsnr observer and eval."""
        obs, needs_eval, needs_cost = resolve_outputs(["sensitivity"])
        assert obs == {"qsnr"}
        assert needs_eval is True
        assert needs_cost is False

    def test_mse_output(self):
        """MSE output: needs mse observer, no eval."""
        obs, needs_eval, needs_cost = resolve_outputs(["mse"])
        assert obs == {"mse"}
        assert needs_eval is False
        assert needs_cost is False

    def test_hierarchical_output(self):
        """Hierarchical output: qsnr + mse observers, needs_eval."""
        obs, needs_eval, needs_cost = resolve_outputs(["hierarchical"])
        assert obs == {"qsnr", "mse"}
        assert needs_eval is True
        assert needs_cost is False

    def test_unknown_key_raises_value_error(self):
        """Unknown output key raises ValueError with valid keys listed."""
        with pytest.raises(ValueError) as excinfo:
            resolve_outputs(["nonexistent_key"])
        assert "nonexistent_key" in str(excinfo.value)
        # Should mention valid keys
        assert "accuracy" in str(excinfo.value)
        assert "qsnr" in str(excinfo.value)

    def test_multiple_unknown_keys(self):
        """Multiple unknown keys all reported in error."""
        with pytest.raises(ValueError) as excinfo:
            resolve_outputs(["bad1", "bad2"])
        assert "bad1" in str(excinfo.value)
        assert "bad2" in str(excinfo.value)

    def test_mixed_known_and_unknown(self):
        """Mixed known + unknown keys raises ValueError."""
        with pytest.raises(ValueError):
            resolve_outputs(["accuracy", "nope"])


class TestPresets:
    """Test PRESETS definitions."""

    def test_default_preset_length(self):
        """Default preset has exactly 2 keys."""
        assert len(PRESETS["default"]) == 2
        assert PRESETS["default"] == ["accuracy", "qsnr"]

    def test_all_preset_contains_all_keys(self):
        """All preset contains every key from _OUTPUT_SPEC."""
        assert set(PRESETS["all"]) == set(_OUTPUT_SPEC.keys())

    def test_all_preset_count(self):
        """All preset contains every key from _OUTPUT_SPEC."""
        assert len(PRESETS["all"]) == len(_OUTPUT_SPEC)


class TestOutputSpec:
    """Test _OUTPUT_SPEC structural invariants."""

    VALID_OBSERVER_KEYS = {"qsnr", "mse", "histogram", "distribution", "fit"}

    def test_every_key_has_required_fields(self):
        """Each output spec has observers (list) and needs_eval (bool)."""
        for key, spec in _OUTPUT_SPEC.items():
            assert "observers" in spec, f"{key} missing 'observers'"
            assert isinstance(spec["observers"], list), f"{key}.observers must be list"
            assert "needs_eval" in spec, f"{key} missing 'needs_eval'"
            assert isinstance(spec["needs_eval"], bool), f"{key}.needs_eval must be bool"

    def test_all_observer_strings_are_valid(self):
        """Every observer string in _OUTPUT_SPEC is a known key."""
        for key, spec in _OUTPUT_SPEC.items():
            for obs in spec["observers"]:
                assert obs in self.VALID_OBSERVER_KEYS, (
                    f"Unknown observer {obs!r} in output key {key!r}. "
                    f"Valid: {sorted(self.VALID_OBSERVER_KEYS)}"
                )

    def test_needs_cost_only_on_cost(self):
        """Only 'cost' output has needs_cost=True."""
        for key, spec in _OUTPUT_SPEC.items():
            if key == "cost":
                assert spec.get("needs_cost", False) is True
            else:
                assert spec.get("needs_cost", False) is False

    def test_empty_observers_lists(self):
        """Outputs with no observers are explicit about it."""
        empty_obs_keys = [k for k, v in _OUTPUT_SPEC.items() if not v["observers"]]
        assert "accuracy" in empty_obs_keys
        assert "pot_delta" in empty_obs_keys
        assert "cost" in empty_obs_keys
        assert "pot_delta_bar" in empty_obs_keys
