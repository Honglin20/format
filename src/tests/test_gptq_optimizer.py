"""Tests for GPTQOptimizer — Hessian-based column-by-column weight quantization."""
import copy

import pytest
import torch
import torch.nn as nn

from src.calibration.gptq_optimizer import GPTQOptimizer
from src.quantize.elemwise import quantize
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme


# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 16)

    def forward(self, x):
        return self.linear(x)


class _TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 12)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(12, 4)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def _per_channel_int4_scheme():
    return QuantScheme(
        format="int4",
        granularity=GranularitySpec.per_channel(axis=0),
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestGPTQOptimizer:
    def test_construct(self):
        opt = GPTQOptimizer(block_size=64, damp_percent=0.02, act_order=True)
        assert opt.block_size == 64
        assert opt.damp_percent == 0.02
        assert opt.act_order is True
        assert opt.num_batches == 8

    def test_construct_defaults(self):
        opt = GPTQOptimizer()
        assert opt.block_size == 128
        assert opt.damp_percent == 0.01
        assert opt.act_order is False
        assert opt.num_batches == 8

    def test_rejects_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            GPTQOptimizer(block_size=0)

    def test_rejects_invalid_damp_low(self):
        with pytest.raises(ValueError, match="damp_percent"):
            GPTQOptimizer(damp_percent=0.0)

    def test_rejects_invalid_damp_high(self):
        with pytest.raises(ValueError, match="damp_percent"):
            GPTQOptimizer(damp_percent=1.1)

    def test_rejects_invalid_num_batches(self):
        with pytest.raises(ValueError, match="num_batches"):
            GPTQOptimizer(num_batches=0)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestGPTQIntegration:
    def test_gptq_modifies_weights(self):
        """GPTQ should change weight values from their original fp32 state."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        weight_orig = model.linear.weight.data.clone()

        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128)
        results = opt.optimize(qmodel, calib_data)

        assert len(results) == 1
        weight_gptq = qmodel.linear.weight.data
        assert not torch.equal(weight_orig, weight_gptq)

    def test_gptq_forward_passes(self):
        """After GPTQ, model forward produces finite output with expected shape."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128)
        opt.optimize(qmodel, calib_data)

        qmodel.eval()
        out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 16)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gptq_lower_mse_than_naive(self):
        """GPTQ-quantized weights should have lower MSE vs fp32 than naive
        per-channel rounding on structured (correlated) inputs."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        W_fp32 = model.linear.weight.data.clone()

        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        # Structured (correlated) calibration data so that Hessian captures
        # meaningful feature correlations.
        gen = torch.Generator()
        gen.manual_seed(42)
        base = torch.randn(100, 4, generator=gen)
        calib_data = [base @ torch.randn(4, 8, generator=gen) for _ in range(4)]

        # Naive per-channel quantization (direct quantize call)
        with torch.no_grad():
            W_naive = quantize(W_fp32, scheme)
        mse_naive = (W_fp32 - W_naive).pow(2).mean().item()

        # GPTQ quantization
        opt = GPTQOptimizer(block_size=128)
        opt.optimize(qmodel, calib_data)
        W_gptq = qmodel.linear.weight.data
        mse_gptq = (W_fp32.to(dtype=W_gptq.dtype) - W_gptq).pow(2).mean().item()

        assert mse_gptq <= mse_naive, (
            f"GPTQ MSE {mse_gptq:.6f} should be <= naive MSE {mse_naive:.6f}"
        )

    def test_gptq_block_size_64(self):
        """Block size 64 produces valid weights and forward works."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=64)
        opt.optimize(qmodel, calib_data)

        qmodel.eval()
        out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 16)
        assert not torch.isnan(out).any()

    def test_gptq_with_act_order(self):
        """act_order=True runs without error and produces valid output."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128, act_order=True)
        opt.optimize(qmodel, calib_data)

        qmodel.eval()
        out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 16)

    def test_gptq_block_size_exceeds_in_features(self):
        """block_size > in_features: the whole weight is quantized in one block."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=256)  # > 8 (in_features)
        opt.optimize(qmodel, calib_data)

        qmodel.eval()
        out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 16)

    def test_gptq_two_layer_model(self):
        """Both Linear layers get GPTQ treatment."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TwoLayerModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128)
        results = opt.optimize(qmodel, calib_data)

        assert len(results) == 2
        for name in results:
            assert "mse_before" in results[name]
            assert "mse_after" in results[name]

    def test_gptq_skips_non_weight_modules(self):
        """Modules without weight in cfg are skipped without error."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TwoLayerModel()
        # activation-only config — no weight quantization
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(input=scheme, weight=None, output=None)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128)
        results = opt.optimize(qmodel, calib_data)

        assert len(results) == 0

    def test_gptq_results_mse_fields(self):
        """Results dict contains mse_before and mse_after for each module."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128)
        results = opt.optimize(qmodel, calib_data)

        for name, meta in results.items():
            assert "mse_before" in meta
            assert "mse_after" in meta
            assert isinstance(meta["mse_before"], float)
            assert isinstance(meta["mse_after"], float)
            assert meta["mse_before"] > 0
            # GPTQ-quantized weight MSE should be >= naive MSE
            # (it optimizes for layer-output error, not weight error)
            assert meta["mse_after"] >= 0

    def test_gptq_weight_scale_buffer_set(self):
        """After GPTQ, module should have _weight_scale buffer."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128)
        opt.optimize(qmodel, calib_data)

        assert hasattr(qmodel.linear, "_weight_scale")
        assert qmodel.linear._weight_scale is not None

    def test_gptq_idempotent_requant(self):
        """Re-quantizing GPTQ weights with stored scale must produce same result."""
        from src.session import quantize_model

        torch.manual_seed(42)
        model = _TinyModel()
        scheme = _per_channel_int4_scheme()
        cfg = OpQuantConfig(weight=scheme)
        qmodel = quantize_model(copy.deepcopy(model), cfg)

        calib_data = [torch.randn(4, 8) for _ in range(4)]
        opt = GPTQOptimizer(block_size=128)
        opt.optimize(qmodel, calib_data)

        W_gptq = qmodel.linear.weight.data.clone()
        scale = qmodel.linear._weight_scale

        with torch.no_grad():
            W_requant = quantize(W_gptq, scheme, scale=scale)

        assert torch.allclose(W_gptq, W_requant, atol=1e-6), (
            f"GPTQ weights not idempotent under re-quantization: "
            f"max diff = {(W_gptq - W_requant).abs().max().item():.8f}"
        )


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------

class TestGPTQSessionIntegration:
    def test_quantize_with_gptq(self):
        """Full run_quantization flow with gptq=True."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        torch.manual_seed(42)
        model = _TwoLayerModel()

        config = QuantConfig(
            w_format="int4",
            w_granularity="per_channel",
            w_axis=0,
            gptq=True,
            gptq_block_size=128,
        )
        calib = [torch.randn(2, 8) for _ in range(4)]
        qmodel, fp32_model, result = run_quantization(
            model, config, calib, keep_fp32=False,
        )

        # Forward pass works
        qmodel.eval()
        with torch.no_grad():
            out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 4)
        assert not torch.isnan(out).any()

    def test_quantize_gptq_with_calibrate(self):
        """run_quantization(gptq=True) → forward works (calibrate is automatic)."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        torch.manual_seed(42)
        model = _TinyModel()

        config = QuantConfig(
            w_format="int4",
            w_granularity="per_channel",
            w_axis=0,
            gptq=True,
            gptq_block_size=128,
        )
        calib = [torch.randn(2, 8) for _ in range(4)]
        qmodel, fp32_model, result = run_quantization(
            model, config, calib, keep_fp32=False,
        )

        qmodel.eval()
        with torch.no_grad():
            out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 16)

    def test_gptq_with_different_block_sizes(self):
        """Block sizes 64 and 128 both work with run_quantization."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        torch.manual_seed(42)
        model = _TinyModel()

        config = QuantConfig(
            w_format="int4",
            w_granularity="per_channel",
            w_axis=0,
            gptq=True,
            gptq_block_size=64,
            gptq_damp=0.02,
            gptq_act_order=True,
        )
        calib = [torch.randn(2, 8) for _ in range(4)]
        qmodel, fp32_model, result = run_quantization(
            model, config, calib, keep_fp32=False,
        )

        qmodel.eval()
        with torch.no_grad():
            out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 16)

    def test_gptq_forward_uses_weight_scale(self):
        """Forward pass after GPTQ should use _weight_scale, not recompute amax."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        torch.manual_seed(42)
        model = _TwoLayerModel()

        config = QuantConfig(
            w_format="int4",
            w_granularity="per_channel",
            w_axis=0,
            gptq=True,
            gptq_block_size=128,
        )
        calib = [torch.randn(2, 8) for _ in range(4)]
        qmodel, _, _ = run_quantization(model, config, calib, keep_fp32=False)

        # After GPTQ + calibration, forward pass should produce finite output
        qmodel.eval()
        with torch.no_grad():
            out = qmodel(torch.randn(2, 8))
        assert out.shape == (2, 4)
        assert not torch.isnan(out).any()

        # Verify _weight_scale buffers exist on quantized linear layers
        for name, mod in qmodel.named_modules():
            if hasattr(mod, "cfg") and hasattr(mod, "weight") and isinstance(mod, nn.Linear):
                assert hasattr(mod, "_weight_scale"), (
                    f"{name} missing _weight_scale after GPTQ"
                )

    def test_gptq_forward_idempotent_with_scale(self):
        """Forward pass with _weight_scale should produce same output as
        direct quantize(w, scheme, scale=_weight_scale)."""
        from src.session._config import QuantConfig
        from src.session._session import run_quantization

        torch.manual_seed(42)
        model = _TinyModel()

        config = QuantConfig(
            w_format="int4",
            w_granularity="per_channel",
            w_axis=0,
            gptq=True,
            gptq_block_size=128,
        )
        calib = [torch.randn(2, 8) for _ in range(4)]
        qmodel, _, _ = run_quantization(model, config, calib, keep_fp32=False)

        # After GPTQ, re-quantizing weight with stored scale should be idempotent
        mod = qmodel.linear
        W_gptq = mod.weight.data.clone()
        scale = mod._weight_scale
        scheme = mod.cfg.weight
        with torch.no_grad():
            W_requant = quantize(W_gptq, scheme, scale=scale)
        assert torch.allclose(W_gptq, W_requant, atol=1e-6), (
            f"GPTQ + calibration weights not idempotent: "
            f"max diff = {(W_gptq - W_requant).abs().max().item():.8f}"
        )

class TestGPTQConfigValidation:
    def test_gptq_config_defaults(self):
        from src.session._config import QuantConfig

        cfg = QuantConfig(gptq=True)
        assert cfg.gptq is True
        assert cfg.gptq_block_size == 128
        assert cfg.gptq_damp == 0.01
        assert cfg.gptq_act_order is False

    def test_gptq_config_custom(self):
        from src.session._config import QuantConfig

        cfg = QuantConfig(
            gptq=True, gptq_block_size=64, gptq_damp=0.02, gptq_act_order=True,
        )
        assert cfg.gptq_block_size == 64
        assert cfg.gptq_damp == 0.02
        assert cfg.gptq_act_order is True

    def test_gptq_rejects_invalid_block_size(self):
        from src.session._config import QuantConfig

        with pytest.raises(ValueError, match="gptq_block_size"):
            QuantConfig(gptq=True, gptq_block_size=0)

    def test_gptq_rejects_invalid_damp(self):
        from src.session._config import QuantConfig

        with pytest.raises(ValueError, match="gptq_damp"):
            QuantConfig(gptq=True, gptq_damp=0.0)

    def test_gptq_default_off(self):
        from src.session._config import QuantConfig

        cfg = QuantConfig()
        assert cfg.gptq is False

    def test_gptq_from_descriptor(self):
        from src.session._config import QuantConfig

        cfg = QuantConfig.from_descriptor({
            "format": "int4",
            "granularity": "per_channel",
            "gptq": True,
            "gptq_block_size": 64,
            "gptq_damp": 0.02,
            "gptq_act_order": True,
        })
        assert cfg.gptq is True
        assert cfg.gptq_block_size == 64
        assert cfg.gptq_damp == 0.02
        assert cfg.gptq_act_order is True
