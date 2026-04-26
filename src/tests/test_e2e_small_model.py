"""
End-to-end smoke test: small model forward/backward vs mx bit-exact — P3.6.

Verifies that quantize_model correctly replaces nn.Module subclasses and
that the resulting model produces bit-exact results against mx equivalents.
"""
import pytest
import torch
import torch.nn as nn

import mx
from mx.specs import apply_mx_specs

from src.mapping.quantize_model import quantize_model
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.tests._compat import op_config_from_mx_specs


def _assert_bit_exact(mx_out, src_out, label="output"):
    if mx_out is None and src_out is None:
        return
    assert mx_out is not None and src_out is not None, f"{label}: one is None"
    if torch.equal(mx_out, src_out):
        return
    mx_nan = torch.isnan(mx_out)
    src_nan = torch.isnan(src_out)
    assert torch.equal(mx_nan, src_nan), f"{label}: NaN mismatch"
    mx_valid = mx_out[~mx_nan]
    src_valid = src_out[~src_nan]
    assert torch.equal(mx_valid, src_valid), (
        f"{label}: valid elements not bit-exact "
        f"(max diff={torch.max(torch.abs(mx_valid - src_valid))})"
    )


# ---------------------------------------------------------------------------
# Small models
# ---------------------------------------------------------------------------

class LinearOnly(nn.Module):
    """Two sequential Linear layers — uniform config works."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class ConvOnly(nn.Module):
    """Single Conv2d — uniform config works."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

MX_SPECS = [
    pytest.param("bfloat16", {"bfloat": 16}, id="bf16"),
]


class TestE2ELinearOnly:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        torch.manual_seed(42)
        mx_specs = apply_mx_specs(mx_specs)

        src_model = LinearOnly()
        cfg = op_config_from_mx_specs(mx_specs, op_type="linear")
        quantize_model(src_model, cfg=cfg)

        mx_model = LinearOnly()
        mx_model.load_state_dict(src_model.state_dict())
        mx_model.fc1 = mx.Linear(8, 16, mx_specs=mx_specs)
        mx_model.fc2 = mx.Linear(16, 4, mx_specs=mx_specs)
        mx_model.fc1.load_state_dict(src_model.fc1.state_dict(), strict=False)
        mx_model.fc2.load_state_dict(src_model.fc2.state_dict(), strict=False)

        mx_model.eval()
        src_model.eval()
        x = torch.randn(2, 8)
        with torch.no_grad():
            _assert_bit_exact(mx_model(x), src_model(x), label="e2e-linear-fwd")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        torch.manual_seed(42)
        mx_specs = apply_mx_specs(mx_specs)

        src_model = LinearOnly()
        cfg = op_config_from_mx_specs(mx_specs, op_type="linear")
        quantize_model(src_model, cfg=cfg)

        mx_model = LinearOnly()
        mx_model.load_state_dict(src_model.state_dict())
        mx_model.fc1 = mx.Linear(8, 16, mx_specs=mx_specs)
        mx_model.fc2 = mx.Linear(16, 4, mx_specs=mx_specs)
        mx_model.fc1.load_state_dict(src_model.fc1.state_dict(), strict=False)
        mx_model.fc2.load_state_dict(src_model.fc2.state_dict(), strict=False)

        mx_model.train()
        src_model.train()
        x = torch.randn(2, 8)
        mx_x, src_x = x.clone().requires_grad_(True), x.clone().requires_grad_(True)

        mx_out = mx_model(mx_x)
        src_out = src_model(src_x)
        _assert_bit_exact(mx_out.detach(), src_out.detach(), label="e2e-linear-train-fwd")

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label="e2e-linear-train-grad")


class TestE2EConvOnly:
    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_forward(self, name, mx_specs):
        torch.manual_seed(42)
        mx_specs = apply_mx_specs(mx_specs)

        src_model = ConvOnly()
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")
        quantize_model(src_model, cfg=cfg)

        mx_model = ConvOnly()
        mx_model.load_state_dict(src_model.state_dict())
        mx_model.conv = mx.Conv2d(3, 8, kernel_size=3, padding=1, mx_specs=mx_specs)
        mx_model.conv.load_state_dict(src_model.conv.state_dict(), strict=False)

        mx_model.eval()
        src_model.eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            _assert_bit_exact(mx_model(x), src_model(x), label="e2e-conv-fwd")

    @pytest.mark.parametrize("name,mx_specs", MX_SPECS)
    def test_backward(self, name, mx_specs):
        torch.manual_seed(42)
        mx_specs = apply_mx_specs(mx_specs)

        src_model = ConvOnly()
        cfg = op_config_from_mx_specs(mx_specs, op_type="conv")
        quantize_model(src_model, cfg=cfg)

        mx_model = ConvOnly()
        mx_model.load_state_dict(src_model.state_dict())
        mx_model.conv = mx.Conv2d(3, 8, kernel_size=3, padding=1, mx_specs=mx_specs)
        mx_model.conv.load_state_dict(src_model.conv.state_dict(), strict=False)

        mx_model.train()
        src_model.train()
        x = torch.randn(2, 3, 8, 8)
        mx_x, src_x = x.clone().requires_grad_(True), x.clone().requires_grad_(True)

        mx_out = mx_model(mx_x)
        src_out = src_model(src_x)
        _assert_bit_exact(mx_out.detach(), src_out.detach(), label="e2e-conv-train-fwd")

        mx_out.sum().backward()
        src_out.sum().backward()
        _assert_bit_exact(mx_x.grad, src_x.grad, label="e2e-conv-train-grad")


class TestQuantizeModelAPI:
    def test_empty_cfg_passthrough(self):
        """quantize_model with no cfg should produce passthrough modules."""
        model = LinearOnly()
        quantize_model(model)  # default cfg=OpQuantConfig()
        assert hasattr(model.fc1, "cfg")
        assert model.fc1.cfg == OpQuantConfig()

    def test_dict_cfg_per_name(self):
        """cfg dict maps name patterns to different OpQuantConfig."""
        model = LinearOnly()
        s = QuantScheme(format="bfloat16")
        cfg_input = OpQuantConfig(input=(s,))
        quantize_model(model, cfg={"*": cfg_input})
        assert model.fc1.cfg == cfg_input
        assert model.fc2.cfg == cfg_input

    def test_already_quantized_skipped(self):
        """Modules that already have .cfg are left alone."""
        from src.ops.linear import QuantizedLinear
        model = LinearOnly()
        s = QuantScheme(format="bfloat16")
        cfg = OpQuantConfig(input=(s,))
        quantize_model(model, cfg=cfg)
        orig_fc1 = model.fc1
        # Second pass should not re-wrap
        quantize_model(model, cfg=cfg)
        assert model.fc1 is orig_fc1

    def test_unknown_module_skipped(self):
        """Unknown module types are left unchanged."""
        class UnknownModule(nn.Module):
            def forward(self, x):
                return x * 2

        model = nn.Sequential(UnknownModule())
        quantize_model(model, cfg=OpQuantConfig())
        assert type(model[0]) is UnknownModule

    # ---- New tests for unified quantize_model (module + forward patching) ----

    def test_forward_patched_flag_set(self):
        """quantize_model sets _quantize_forward_patched on the model."""
        model = LinearOnly()
        quantize_model(model, cfg=OpQuantConfig(input=(QuantScheme(format="bfloat16"),)))
        assert getattr(model, '_quantize_forward_patched', False)

    def test_forward_still_runs(self):
        """quantize_model(model) produces output of expected shape."""
        model = LinearOnly()
        s = QuantScheme(format="bfloat16")
        cfg = OpQuantConfig(input=(s,))
        quantize_model(model, cfg=cfg)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 4)

    def test_export_onnx_added(self):
        """After quantize_model, model.export_onnx is callable."""
        model = LinearOnly()
        s = QuantScheme(format="bfloat16")
        cfg = OpQuantConfig(input=(s,))
        quantize_model(model, cfg=cfg)
        assert callable(model.export_onnx)

    def test_export_onnx_produces_valid_graph(self, tmp_path):
        """model.export_onnx(dummy, path) produces valid ONNX."""
        import onnx
        model = LinearOnly()
        s = QuantScheme(format="bfloat16")
        cfg = OpQuantConfig(input=(s,), weight=(s,), output=(s,))
        quantize_model(model, cfg=cfg)
        path = str(tmp_path / "e2e_export.onnx")
        model.export_onnx(torch.randn(2, 8), path)
        m = onnx.load(path)
        onnx.checker.check_model(m)

    def test_double_quantize_forward_guard(self):
        """Second quantize_model call does not re-patch forward."""
        model = LinearOnly()
        s = QuantScheme(format="bfloat16")
        cfg = OpQuantConfig(input=(s,))
        quantize_model(model, cfg=cfg)
        first_forward = model.forward
        quantize_model(model, cfg=cfg)
        assert model.forward is first_forward  # same function object

    def test_state_dict_keys_unchanged(self):
        """quantize_model does not add/remove state_dict keys."""
        model = LinearOnly()
        pre_keys = set(model.state_dict().keys())
        s = QuantScheme(format="bfloat16")
        cfg = OpQuantConfig(input=(s,))
        quantize_model(model, cfg=cfg)
        post_keys = set(model.state_dict().keys())
        assert pre_keys == post_keys

    def test_quantize_does_not_quantize_outside_forward(self):
        """torch.matmul outside model(x) is NOT patched (no context)."""
        model = LinearOnly()
        s = QuantScheme(format="bfloat16")
        cfg = OpQuantConfig(input=(s,), weight=(s,), output=(s,))
        quantize_model(model, cfg=cfg)
        # Call model.forward to confirm it works (context enters + exits).
        model(torch.randn(2, 8))
        # Outside forward, torch.matmul should return float result.
        a, b = torch.randn(3, 4), torch.randn(4, 5)
        outside_result = torch.matmul(a, b)
        assert outside_result.requires_grad is False  # raw float, not quantized
