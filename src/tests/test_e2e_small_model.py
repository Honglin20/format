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
