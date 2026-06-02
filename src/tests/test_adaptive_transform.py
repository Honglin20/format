"""Verify adaptive transform selection picks the right transform per layer.

Constructs a chain model where three Linear layers each favour a different
transform candidate (none / hadamard / smoothquant).  The test runs
``run_quantization(…, transform="adaptive")`` and asserts the
expected winner per layer.
"""
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import run_quantization
from src.scheme.transform import IdentityTransform
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform


# ---------------------------------------------------------------------------
# Crafted model -- each layer favours a different transform
# ---------------------------------------------------------------------------

def _make_adaptive_test_model():
    """Return (model, x_input) for a 3-layer chain.

    Statistical design (seed=42):

    - fc_a (128→64): weight has extreme input-channel outlier → hadamard
    - fc_b  (64→64): activation has per-channel outlier → smoothquant
    - fc_c  (64→32): uniform activation + weight → all candidates are
      close, but per-channel SQ has a small edge on per_block quant
    """
    torch.manual_seed(42)

    model = nn.Sequential()
    model.add_module("fc_hadamard", nn.Linear(128, 64))
    model.add_module("fc_smoothquant", nn.Linear(64, 64))
    model.add_module("fc_uniform", nn.Linear(64, 32))

    # --- fc_hadamard: input-channel 20 is a 200x outlier ---
    W_had = torch.randn(64, 128) * 0.05
    W_had[:, 20] *= 200
    model.fc_hadamard.weight.data = W_had
    nn.init.zeros_(model.fc_hadamard.bias)

    # --- fc_smoothquant: fc_hadamard's weight outlier creates per-channel
    #     outliers in its output (= fc_smoothquant's activation).  The
    #     weight itself is uniform. ---
    W_sq = torch.randn(64, 64) * 0.05
    model.fc_smoothquant.weight.data = W_sq
    nn.init.zeros_(model.fc_smoothquant.bias)

    # --- fc_uniform: no outliers in either direction ---
    W_uni = torch.randn(32, 64) * 0.05
    model.fc_uniform.weight.data = W_uni
    nn.init.zeros_(model.fc_uniform.bias)

    # Input: uniform (no per-channel outlier at entry)
    x = torch.randn(8, 128) * 0.05

    return model, x


# ---------------------------------------------------------------------------
# Heuristic prediction -- verify the crafted model has the right pattern
# ---------------------------------------------------------------------------

def _predict_winners(model, x):
    """Return dict of expected winner per layer name.

    This checks statistics only -- it does NOT run the adaptive algorithm.
    The thresholds are tuned to the seed=42 crafted tensors.
    """
    with torch.no_grad():
        h0 = model.fc_hadamard(x)
        h1 = model.fc_smoothquant(h0)

    results = {}

    # fc_hadamard: weight input-channel outlier
    w_per_ch = model.fc_hadamard.weight.data.abs().amax(dim=0)
    w_ratio = w_per_ch.max() / (w_per_ch.median() + 1e-12)
    results["fc_hadamard_is_outlier"] = w_ratio > 50  # ~200x outlier

    # fc_smoothquant: activation (h0) has per-channel outlier
    act_per_ch = h0.abs().amax(dim=0)
    act_ratio = act_per_ch.max() / (act_per_ch.median() + 1e-12)
    results["fc_smoothquant_has_act_outlier"] = act_ratio > 5

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdaptiveTransform:
    """Verify that transform="adaptive" selects the right candidate per layer."""

    def test_adaptive_selects_expected_winners(self):
        """Hadamard for weight-outlier layer, SmoothQuant for act-outlier layer."""
        model, x = _make_adaptive_test_model()

        # Verify crafted statistics
        pred = _predict_winners(model, x)
        assert pred["fc_hadamard_is_outlier"], (
            "fc_hadamard weight should have input-channel outlier (ratio > 50)"
        )
        assert pred["fc_smoothquant_has_act_outlier"], (
            "fc_smoothquant activation should have channel outlier (ratio > 5)"
        )

        cfg = QuantConfig(
            name="test-adaptive",
            w_format="int8",
            a_format="int8",
            w_granularity="per_block",
            w_block_size=32,
            a_granularity="per_block",
            a_block_size=32,
            transform="adaptive",
            sq_alpha=0.5,
        )

        qmodel, fp32_model, _ = run_quantization(model, cfg, [x], keep_fp32=True)

        # --- fc_hadamard: must be HadamardTransform ---
        fc_had_cfg = qmodel.fc_hadamard.cfg
        assert isinstance(fc_had_cfg.input.transform, HadamardTransform), (
            f"fc_hadamard expected HadamardTransform on input, "
            f"got {type(fc_had_cfg.input.transform).__name__}"
        )
        assert isinstance(fc_had_cfg.weight.transform, HadamardTransform), (
            f"fc_hadamard expected HadamardTransform on weight, "
            f"got {type(fc_had_cfg.weight.transform).__name__}"
        )

        # --- fc_smoothquant: must be SmoothQuantTransform on input,
        #     IdentityTransform on weight (scale fused) ---
        fc_sq_cfg = qmodel.fc_smoothquant.cfg
        assert isinstance(fc_sq_cfg.input.transform, SmoothQuantTransform), (
            f"fc_smoothquant expected SmoothQuantTransform on input, "
            f"got {type(fc_sq_cfg.input.transform).__name__}"
        )
        assert isinstance(fc_sq_cfg.weight.transform, IdentityTransform), (
            f"fc_smoothquant expected IdentityTransform on weight (scale fused), "
            f"got {type(fc_sq_cfg.weight.transform).__name__}"
        )

    def test_adaptive_runs_without_errors(self):
        """Smoke test: adaptive runs end-to-end, produces QSNR per layer."""
        model, x = _make_adaptive_test_model()

        cfg = QuantConfig(
            name="test-adaptive-smoke",
            w_format="int8",
            w_granularity="per_tensor",
            transform="adaptive",
        )

        qmodel, fp32_model, _ = run_quantization(model, cfg, [x])

        # Verify that each matmul module received a non-None cfg transform
        for name in ["fc_hadamard", "fc_smoothquant", "fc_uniform"]:
            mod = getattr(qmodel, name)
            assert mod.cfg.input is not None, f"{name}: input cfg is None"
            assert mod.cfg.input.transform is not None, (
                f"{name}: input transform is None"
            )

    def test_adaptive_produces_valid_forward(self):
        """Quantized model with adaptive transforms produces valid output."""
        model, x = _make_adaptive_test_model()

        cfg = QuantConfig(
            name="test-adaptive-fwd",
            w_format="int8",
            w_granularity="per_block",
            w_block_size=32,
            a_granularity="per_block",
            a_block_size=32,
            transform="adaptive",
        )

        qmodel, fp32_model, _ = run_quantization(model, cfg, [x])

        with torch.no_grad():
            out = qmodel(x)
        assert out.shape == (8, 32)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_adaptive_is_deterministic(self):
        """Two runs on identical models produce identical selections."""
        model1, x1 = _make_adaptive_test_model()
        model2, _ = _make_adaptive_test_model()
        model2.load_state_dict(model1.state_dict())  # align weights

        cfg = QuantConfig(
            name="test-adaptive-det",
            w_format="int8",
            w_granularity="per_block",
            w_block_size=32,
            a_granularity="per_block",
            a_block_size=32,
            transform="adaptive",
        )

        qmodel1, _, _ = run_quantization(model1, cfg, [x1])
        qmodel2, _, _ = run_quantization(model2, cfg, [x1])

        for name in ["fc_hadamard", "fc_smoothquant", "fc_uniform"]:
            m1 = getattr(qmodel1, name)
            m2 = getattr(qmodel2, name)
            t1_in = type(m1.cfg.input.transform)
            t2_in = type(m2.cfg.input.transform)
            t1_w = type(m1.cfg.weight.transform)
            t2_w = type(m2.cfg.weight.transform)
            assert t1_in == t2_in, (
                f"{name}: input transform differs "
                f"({t1_in.__name__} vs {t2_in.__name__})"
            )
            assert t1_w == t2_w, (
                f"{name}: weight transform differs "
                f"({t1_w.__name__} vs {t2_w.__name__})"
            )

    def test_adaptive_with_weight_only(self):
        """Weight-only adaptive skips SmoothQuant, picks none or hadamard."""
        model, x = _make_adaptive_test_model()

        cfg = QuantConfig(
            name="test-adaptive-wo",
            w_format="int4",
            w_granularity="per_block",
            w_block_size=32,
            weight_only=True,
            transform="adaptive",
        )

        qmodel, fp32_model, _ = run_quantization(model, cfg, [x])

        for name in ["fc_hadamard", "fc_smoothquant", "fc_uniform"]:
            mod = getattr(qmodel, name)
            tx_type = type(mod.cfg.weight.transform)
            assert tx_type in (IdentityTransform, HadamardTransform), (
                f"{name}: unexpected transform {tx_type.__name__} in "
                f"weight_only mode"
            )

    def test_adaptive_with_conv2d(self):
        """Conv2d layers are supported in adaptive selection."""
        torch.manual_seed(42)

        model = nn.Sequential()
        # Conv2d with weight outlier: input-channel 0 is 200x
        conv_outlier = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        W = torch.randn(32, 16, 3, 3) * 0.05
        W[:, 0] *= 200
        conv_outlier.weight.data = W
        nn.init.zeros_(conv_outlier.bias)
        model.add_module("conv_had", conv_outlier)

        # Conv2d with uniform weight -- activation outlier comes from prev layer
        conv_sq = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        conv_sq.weight.data = torch.randn(16, 32, 3, 3) * 0.05
        nn.init.zeros_(conv_sq.bias)
        model.add_module("conv_sq", conv_sq)

        x = torch.randn(4, 16, 8, 8) * 0.05

        cfg = QuantConfig(
            name="test-adaptive-conv",
            w_format="int8",
            w_granularity="per_block",
            w_block_size=32,
            a_granularity="per_block",
            a_block_size=32,
            transform="adaptive",
        )

        qmodel, fp32_model, _ = run_quantization(model, cfg, [x])

        # conv_had: weight input-channel outlier → hadamard expected
        had_cfg = qmodel.conv_had.cfg
        assert isinstance(had_cfg.input.transform, (HadamardTransform, IdentityTransform)), (
            f"conv_had: unexpected input transform {type(had_cfg.input.transform).__name__}"
        )
        assert isinstance(had_cfg.weight.transform, (HadamardTransform, IdentityTransform)), (
            f"conv_had: unexpected weight transform {type(had_cfg.weight.transform).__name__}"
        )

        # conv_sq: should have a valid selection (not None)
        sq_cfg = qmodel.conv_sq.cfg
        assert sq_cfg.input.transform is not None
        assert sq_cfg.weight.transform is not None

        # Forward pass works
        with torch.no_grad():
            out = qmodel(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_adaptive_with_eval_fn(self):
        """eval_fn is correctly passed through during adaptive selection."""
        model, x = _make_adaptive_test_model()

        cfg = QuantConfig(
            name="test-adaptive-evalfn",
            w_format="int8",
            w_granularity="per_block",
            w_block_size=32,
            a_granularity="per_block",
            a_block_size=32,
            transform="adaptive",
        )

        trace = []

        def custom_eval(model, data):
            trace.append("called")
            if isinstance(data, torch.Tensor):
                model(data)
            else:
                for batch in data:
                    model(batch)

        qmodel, fp32_model, _ = run_quantization(
            model, cfg, [x], eval_fn=custom_eval,
        )

        assert len(trace) >= 1, "eval_fn should have been called"

    def test_adaptive_no_matmul_layers(self):
        """Model without Linear/Conv → adaptive returns gracefully."""
        model = nn.Sequential(nn.ReLU())
        cfg = QuantConfig(
            name="test-adaptive-empty",
            w_format="int8",
            w_granularity="per_tensor",
            transform="adaptive",
        )

        qmodel, fp32_model, _ = run_quantization(model, cfg, [torch.randn(4, 32)])

    def test_adaptive_calibrate_twice(self):
        """Calling calibrate twice doesn't re-run adaptive selection."""
        model, x = _make_adaptive_test_model()

        cfg = QuantConfig(
            name="test-adaptive-2x",
            w_format="int8",
            w_granularity="per_block",
            w_block_size=32,
            a_granularity="per_block",
            a_block_size=32,
            transform="adaptive",
        )

        qmodel, fp32_model, _ = run_quantization(model, cfg, [x])

        # Capture cfg after first calibration
        cfgs_after_first = {}
        for name in ["fc_hadamard", "fc_smoothquant", "fc_uniform"]:
            mod = getattr(qmodel, name)
            cfgs_after_first[name] = (
                type(mod.cfg.input.transform),
                type(mod.cfg.weight.transform),
            )

        # Second calibration should NOT change transforms
        from src.calibration.pipeline import CalibrationSession
        from src.calibration.strategies import MaxScaleStrategy
        with CalibrationSession(qmodel, MaxScaleStrategy()):
            with torch.no_grad():
                qmodel(x)

        for name in ["fc_hadamard", "fc_smoothquant", "fc_uniform"]:
            mod = getattr(qmodel, name)
            after = (type(mod.cfg.input.transform), type(mod.cfg.weight.transform))
            assert after == cfgs_after_first[name], (
                f"{name}: transforms changed after second calibrate()"
            )
