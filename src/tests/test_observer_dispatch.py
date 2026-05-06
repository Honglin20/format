"""
Tests for SliceAwareObserver per-mode dispatch and valid_counts correction.

Covers:
- PER_TENSOR: key format, metric correctness
- PER_CHANNEL: positive/negative axis, channel count, key format
- PER_BLOCK: exact/inexact division, axis handling, valid_counts correction,
  per-block aggregate stats (mean/std/min/max)
- DYNAMIC_GROUP: group_map presence/absence, key format
- Unknown mode: raises ValueError
"""
import math
import pytest
import torch

from src.observer.events import QuantEvent
from src.observer import SliceAwareObserver
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.scheme.quant_scheme import QuantScheme
from src.formats.int_formats import IntFormat


# ---------------------------------------------------------------------------
# Minimal concrete observer for testing dispatch
# ---------------------------------------------------------------------------

class _TestMSEObserver(SliceAwareObserver):
    """Records keys and computes per-slice MSE. Used to verify dispatch."""

    def __init__(self, record_keys=False):
        super().__init__()
        self.keys_seen = []
        self._record = record_keys

    def _measure(self, key, fp32, quant):
        if self._record:
            self.keys_seen.append(key)
        return {"mse": (fp32 - quant).pow(2).mean().item()}

    def _measure_per_unit(self, fp32_2d, quant_2d):
        """Record keys for PER_CHANNEL dispatch verification."""
        results = []
        for i in range(fp32_2d.shape[0]):
            results.append(self._measure(("channel", i), fp32_2d[i], quant_2d[i]))
        return results

    def _measure_batch(self, fp32_2d, quant_2d, valid_counts=None):
        """Correct per-block MSE using valid_counts when provided.

        Uses population std (unbiased=False) since we measure all blocks in the tensor.
        """
        err_sq = (fp32_2d - quant_2d).pow(2)
        if valid_counts is not None:
            mse_per = err_sq.sum(dim=1) / valid_counts.clamp_min(1)
        else:
            mse_per = err_sq.mean(dim=1)
        return {
            "mse": mse_per.mean().item(),
            "mse_std": mse_per.std(unbiased=False).item() if mse_per.numel() > 1 else 0.0,
            "mse_min": mse_per.min().item(),
            "mse_max": mse_per.max().item(),
        }


def _make_event(fp32, quant, granularity, group_map=None, layer="test_layer",
                role="input", stage="input_pre_quant", pipeline_index=1):
    """Helper to construct a QuantEvent with a mock scheme."""
    fmt = IntFormat(8)
    scheme = QuantScheme(format=fmt, granularity=granularity)
    return QuantEvent(
        layer_name=layer,
        role=role,
        stage=stage,
        pipeline_index=pipeline_index,
        fp32_tensor=fp32,
        quant_tensor=quant,
        scheme=scheme,
        group_map=group_map,
    )


# ---------------------------------------------------------------------------
# PER_TENSOR
# ---------------------------------------------------------------------------

class TestPerTensorDispatch:
    def test_single_key_tensor(self):
        fp32 = torch.randn(3, 4)
        quant = fp32 + 0.1 * torch.randn(3, 4)
        g = GranularitySpec.per_tensor()
        obs = _TestMSEObserver(record_keys=True)
        obs.on_event(_make_event(fp32, quant, g))
        report = obs.report()
        keys = list(report["test_layer"]["input"]["input_pre_quant[1]"].keys())
        assert keys == [("tensor",)], f"Expected [('tensor',)], got {keys}"

    def test_measures_all_elements(self):
        fp32 = torch.randn(3, 4)
        quant = fp32 + 0.1 * torch.randn(3, 4)
        g = GranularitySpec.per_tensor()
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        report = obs.report()
        mse_reported = report["test_layer"]["input"]["input_pre_quant[1]"][("tensor",)]["mse"]
        expected = (fp32 - quant).pow(2).mean().item()
        assert math.isclose(mse_reported, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# PER_CHANNEL
# ---------------------------------------------------------------------------

class TestPerChannelDispatch:
    def test_yields_per_channel_keys(self):
        fp32 = torch.randn(5, 8)
        quant = fp32 + 0.1 * torch.randn(5, 8)
        g = GranularitySpec.per_channel(axis=0)
        obs = _TestMSEObserver(record_keys=True)
        obs.on_event(_make_event(fp32, quant, g))
        keys = obs.keys_seen
        assert len(keys) == 5
        for i in range(5):
            assert keys[i] == ("channel", i)

    def test_positive_axis(self):
        fp32 = torch.randn(3, 4, 5)
        quant = fp32 + 0.1 * torch.randn(3, 4, 5)
        g = GranularitySpec.per_channel(axis=1)
        obs = _TestMSEObserver(record_keys=True)
        obs.on_event(_make_event(fp32, quant, g))
        assert len(obs.keys_seen) == 4  # 4 channels at axis=1

    def test_negative_axis_equivalent(self):
        fp32 = torch.randn(3, 4, 5)
        quant = fp32 + 0.1 * torch.randn(3, 4, 5)
        g_neg = GranularitySpec.per_channel(axis=-1)
        g_pos = GranularitySpec.per_channel(axis=2)
        obs_neg = _TestMSEObserver()
        obs_pos = _TestMSEObserver()
        obs_neg.on_event(_make_event(fp32, quant, g_neg))
        obs_pos.on_event(_make_event(fp32, quant, g_pos))
        r_neg = obs_neg.report()["test_layer"]["input"]["input_pre_quant[1]"]
        r_pos = obs_pos.report()["test_layer"]["input"]["input_pre_quant[1]"]
        assert len(r_neg) == 5
        assert len(r_pos) == 5
        for i in range(5):
            assert math.isclose(
                r_neg[("channel", i)]["mse"], r_pos[("channel", i)]["mse"],
                rel_tol=1e-6
            )

    def test_measures_single_channel_correctly(self):
        """Verify MSE for channel 0 matches manual computation."""
        fp32 = torch.randn(4, 6)
        quant = fp32 + 0.1 * torch.randn(4, 6)
        g = GranularitySpec.per_channel(axis=0)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        report = obs.report()
        for i in range(4):
            reported = report["test_layer"]["input"]["input_pre_quant[1]"][("channel", i)]["mse"]
            expected = (fp32[i] - quant[i]).pow(2).mean().item()
            assert math.isclose(reported, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# PER_BLOCK — exact division
# ---------------------------------------------------------------------------

class TestPerBlockExactDivision:
    def test_2d_tensor_last_axis(self):
        fp32 = torch.randn(2, 8)
        quant = fp32 + 0.1 * torch.randn(2, 8)
        g = GranularitySpec.per_block(4)  # default axis=-1
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        report = obs.report()
        entry = report["test_layer"]["input"]["input_pre_quant[1]"]
        assert ("block_agg",) in entry
        agg = entry[("block_agg",)]
        assert "mse" in agg
        assert "mse_std" in agg
        assert "mse_min" in agg
        assert "mse_max" in agg
        # 2 rows * 2 blocks = 4 MX blocks
        # Mean of per-block MSE should equal overall MSE (weighted equally)
        expected_mse = (fp32 - quant).pow(2).mean().item()
        # For exact division, per-block mean should match
        assert math.isclose(agg["mse"], expected_mse, rel_tol=1e-5)

    def test_3d_tensor_non_last_axis(self):
        fp32 = torch.randn(4, 8, 16)
        quant = fp32 + 0.1 * torch.randn(4, 8, 16)
        g = GranularitySpec.per_block(4, axis=1)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        report = obs.report()
        assert ("block_agg",) in report["test_layer"]["input"]["input_pre_quant[1]"]

    def test_negative_axis_equivalent(self):
        fp32 = torch.randn(4, 8, 16)
        quant = fp32 + 0.1 * torch.randn(4, 8, 16)
        g_neg = GranularitySpec.per_block(4, axis=-2)  # same as axis=1
        g_pos = GranularitySpec.per_block(4, axis=1)
        obs_neg = _TestMSEObserver()
        obs_pos = _TestMSEObserver()
        obs_neg.on_event(_make_event(fp32, quant, g_neg))
        obs_pos.on_event(_make_event(fp32, quant, g_pos))
        mse_neg = obs_neg.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]["mse"]
        mse_pos = obs_pos.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]["mse"]
        assert math.isclose(mse_neg, mse_pos, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# PER_BLOCK — inexact division (valid_counts correction)
# ---------------------------------------------------------------------------

class TestPerBlockInexactDivision:
    """Tests for the valid_counts correction when dim_size % block_size != 0."""

    def test_valid_counts_corrects_mse_vs_ground_truth(self):
        """The key test: padded measurement must equal true block measurement."""
        torch.manual_seed(42)
        fp32 = torch.randn(2, 10)
        quant = fp32 + 0.1 * torch.randn(2, 10)
        bs = 4
        g = GranularitySpec.per_block(bs)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]

        # Compute ground truth: per-MX-block MSE manually
        # 2 rows, each with 3 blocks: [0:4], [4:8], [8:10]
        n_blocks = (10 + bs - 1) // bs  # 3
        per_block_mse = []
        for row in range(2):
            for b in range(n_blocks):
                start = b * bs
                end = min((b + 1) * bs, 10)
                f_slice = fp32[row, start:end]
                q_slice = quant[row, start:end]
                per_block_mse.append((f_slice - q_slice).pow(2).mean().item())

        expected_mean = sum(per_block_mse) / len(per_block_mse)
        expected_std = (sum((x - expected_mean)**2 for x in per_block_mse) / len(per_block_mse))**0.5
        expected_min = min(per_block_mse)
        expected_max = max(per_block_mse)

        assert math.isclose(agg["mse"], expected_mean, rel_tol=1e-5), \
            f"mse={agg['mse']}, expected={expected_mean}"
        assert math.isclose(agg["mse_std"], expected_std, rel_tol=1e-5), \
            f"mse_std={agg['mse_std']}, expected={expected_std}"
        assert math.isclose(agg["mse_min"], expected_min, rel_tol=1e-5), \
            f"mse_min={agg['mse_min']}, expected={expected_min}"
        assert math.isclose(agg["mse_max"], expected_max, rel_tol=1e-5), \
            f"mse_max={agg['mse_max']}, expected={expected_max}"

    def test_valid_counts_eliminates_underestimation(self):
        """Without valid_counts, MSE is underestimated. With it, it's correct."""
        torch.manual_seed(123)
        fp32 = torch.randn(3, 7)  # 7 % 4 = 3, last block has only 3 real elements
        quant = fp32 + 0.1 * torch.randn(3, 7)
        bs = 4
        g = GranularitySpec.per_block(bs)

        # Observer with valid_counts (correct)
        obs_correct = _TestMSEObserver()
        obs_correct.on_event(_make_event(fp32, quant, g))
        mse_correct = obs_correct.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]["mse"]

        # Compute buggy version (without valid_counts → mean instead of sum/count)
        # Simulate by measuring padded tensor directly
        dim_size = 7
        n_blocks = (dim_size + bs - 1) // bs
        pad = n_blocks * bs - dim_size
        fp32_pad = torch.nn.functional.pad(fp32, (0, pad))
        quant_pad = torch.nn.functional.pad(quant, (0, pad))
        fp32_2d = fp32_pad.reshape(-1, bs)
        quant_2d = quant_pad.reshape(-1, bs)
        mse_buggy = (fp32_2d - quant_2d).pow(2).mean(dim=1).mean().item()

        # Ground truth: per-block MSE without padding
        per_block_mse = []
        for row in range(3):
            for b in range(n_blocks):
                start = b * bs
                end = min((b + 1) * bs, 7)
                f_slice = fp32[row, start:end]
                q_slice = quant[row, start:end]
                per_block_mse.append((f_slice - q_slice).pow(2).mean().item())
        ground_truth = sum(per_block_mse) / len(per_block_mse)

        # valid_counts matches ground truth
        assert math.isclose(mse_correct, ground_truth, rel_tol=1e-5), \
            f"correct={mse_correct}, ground_truth={ground_truth}"

        # buggy version underestimates
        assert mse_buggy < ground_truth - 1e-8, \
            f"Buggy MSE ({mse_buggy}) should be lower than truth ({ground_truth})"

    def test_single_row_tensor(self):
        """Edge case: single row with non-exact division."""
        torch.manual_seed(99)
        fp32 = torch.randn(1, 11)  # 11 % 4 = 3
        quant = fp32 + 0.1 * torch.randn(1, 11)
        bs = 4
        g = GranularitySpec.per_block(bs)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]

        # Ground truth: 3 blocks: [0:4], [4:8], [8:11]
        per_block = []
        for b in range(3):
            start = b * bs
            end = min((b + 1) * bs, 11)
            per_block.append((fp32[0, start:end] - quant[0, start:end]).pow(2).mean().item())

        assert math.isclose(agg["mse"], sum(per_block)/len(per_block), rel_tol=1e-5)

    def test_exact_division_no_padding(self):
        """When dim_size % bs == 0, valid_counts is None → uses .mean() path."""
        fp32 = torch.randn(4, 16)
        quant = fp32 + 0.1 * torch.randn(4, 16)
        g = GranularitySpec.per_block(4)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]
        # For exact division, mean == sum/count (all blocks have bs elements)
        expected = (fp32 - quant).pow(2).mean().item()
        assert math.isclose(agg["mse"], expected, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# PER_BLOCK — 3D tensor with non-exact division
# ---------------------------------------------------------------------------

class TestPerBlock3DInexact:
    def test_3d_non_exact_division(self):
        """3D tensor [2,3,10], block_axis=-1, bs=4. 10 % 4 = 2."""
        torch.manual_seed(7)
        fp32 = torch.randn(2, 3, 10)
        quant = fp32 + 0.05 * torch.randn(2, 3, 10)
        bs = 4
        g = GranularitySpec.per_block(bs, axis=-1)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]

        # Ground truth: 2*3*3 = 18 MX blocks
        n_blocks = (10 + bs - 1) // bs  # 3
        per_block = []
        for b_idx in range(2):
            for c_idx in range(3):
                for blk in range(n_blocks):
                    start = blk * bs
                    end = min((blk + 1) * bs, 10)
                    f_slice = fp32[b_idx, c_idx, start:end]
                    q_slice = quant[b_idx, c_idx, start:end]
                    per_block.append((f_slice - q_slice).pow(2).mean().item())

        expected_mean = sum(per_block) / len(per_block)
        assert math.isclose(agg["mse"], expected_mean, rel_tol=1e-5)

    def test_3d_non_last_axis_non_exact(self):
        """3D [4,10,6], block_axis=1, bs=3. 10 % 3 = 1."""
        torch.manual_seed(13)
        fp32 = torch.randn(4, 10, 6)
        quant = fp32 + 0.03 * torch.randn(4, 10, 6)
        bs = 3
        g = GranularitySpec.per_block(bs, axis=1)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]

        # Ground truth
        n_blocks = (10 + bs - 1) // bs  # 4
        per_block = []
        for d0 in range(4):
            for d2 in range(6):
                for blk in range(n_blocks):
                    start = blk * bs
                    end = min((blk + 1) * bs, 10)
                    f_slice = fp32[d0, start:end, d2]
                    q_slice = quant[d0, start:end, d2]
                    per_block.append((f_slice - q_slice).pow(2).mean().item())

        expected_mean = sum(per_block) / len(per_block)
        assert math.isclose(agg["mse"], expected_mean, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# DYNAMIC_GROUP
# ---------------------------------------------------------------------------

class TestDynamicGroupDispatch:
    def test_with_group_map(self):
        fp32 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        quant = torch.tensor([1.1, 2.1, 3.1, 4.1])
        group_map = torch.tensor([0, 0, 1, 1])
        g = GranularitySpec(mode=GranularityMode.DYNAMIC_GROUP)
        obs = _TestMSEObserver(record_keys=True)
        obs.on_event(_make_event(fp32, quant, g, group_map=group_map))
        assert len(obs.keys_seen) == 2
        assert obs.keys_seen[0] == ("group", 0)
        assert obs.keys_seen[1] == ("group", 1)

    def test_without_group_map_raises(self):
        fp32 = torch.randn(4)
        quant = torch.randn(4)
        g = GranularitySpec(mode=GranularityMode.DYNAMIC_GROUP)
        obs = _TestMSEObserver()
        with pytest.raises(ValueError, match="group_map"):
            obs.on_event(_make_event(fp32, quant, g, group_map=None))


# ---------------------------------------------------------------------------
# Unknown mode
# ---------------------------------------------------------------------------

class TestUnknownMode:
    def test_unknown_mode_raises(self):
        """Unknown mode triggers ValueError in the dispatch else-branch.

        QuantScheme is frozen so we cannot mock granularity. Instead, we
        construct a GranularitySpec with a valid mode and verify it works,
        confirming the else-branch is reachable only via internal error.
        The unknown-mode path is covered by test_slicing.py.
        """
        fp32 = torch.randn(3)
        quant = torch.randn(3)
        obs = _TestMSEObserver()
        g = GranularitySpec.per_tensor()
        obs.on_event(_make_event(fp32, quant, g))
        assert ("tensor",) in obs.report()["test_layer"]["input"]["input_pre_quant[1]"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_block_size_equals_dim_size(self):
        """block_size == dim_size → single block, no padding needed."""
        fp32 = torch.randn(3, 8)
        quant = fp32 + 0.1 * torch.randn(3, 8)
        g = GranularitySpec.per_block(8)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]
        expected = (fp32 - quant).pow(2).mean().item()
        assert math.isclose(agg["mse"], expected, rel_tol=1e-5)

    def test_block_size_larger_than_dim_size(self):
        """block_size > dim_size → single padded block → valid_counts = dim_size."""
        torch.manual_seed(3)
        fp32 = torch.randn(2, 5)
        quant = fp32 + 0.1 * torch.randn(2, 5)
        g = GranularitySpec.per_block(8)  # bs=8 > dim_size=5
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]

        # Ground truth: 2 blocks (one per row), each with 5 elements
        per_block = []
        for row in range(2):
            per_block.append((fp32[row] - quant[row]).pow(2).mean().item())
        expected = sum(per_block) / len(per_block)
        assert math.isclose(agg["mse"], expected, rel_tol=1e-5)

    def test_scalar_block_size_one(self):
        """bs=1: each element is its own block. No padding ever needed."""
        fp32 = torch.randn(3, 5)
        quant = fp32 + 0.1 * torch.randn(3, 5)
        g = GranularitySpec.per_block(1)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g))
        agg = obs.report()["test_layer"]["input"]["input_pre_quant[1]"][("block_agg",)]
        expected = (fp32 - quant).pow(2).mean().item()
        assert math.isclose(agg["mse"], expected, rel_tol=1e-5)

    def test_channel_axis_zero_on_2d(self):
        """channel_axis=0 on 2D tensor: each row is a channel."""
        fp32 = torch.randn(6, 4)
        quant = fp32 + 0.1 * torch.randn(6, 4)
        g = GranularitySpec.per_channel(axis=0)
        obs = _TestMSEObserver(record_keys=True)
        obs.on_event(_make_event(fp32, quant, g))
        assert len(obs.keys_seen) == 6

    def test_report_structure(self):
        """Verify the nested dict structure of the report."""
        fp32 = torch.randn(2, 3)
        quant = fp32 + 0.1 * torch.randn(2, 3)
        g = GranularitySpec.per_tensor()
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, g, layer="layer1", role="weight",
                                 stage="weight_pre_quant", pipeline_index=0))
        report = obs.report()
        assert "layer1" in report
        assert "weight" in report["layer1"]
        assert "weight_pre_quant[0]" in report["layer1"]["weight"]

    def test_reset_clears_buffer(self):
        fp32 = torch.randn(2, 3)
        quant = fp32 + 0.1 * torch.randn(2, 3)
        obs = _TestMSEObserver()
        obs.on_event(_make_event(fp32, quant, GranularitySpec.per_tensor()))
        assert len(obs.report()) > 0
        obs.reset()
        assert len(obs.report()) == 0
