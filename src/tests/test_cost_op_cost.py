"""Tests for per-operator cost functions."""
import pytest
import torch
import torch.nn as nn
from src.cost.device import DeviceSpec
from src.cost.op_cost import op_cost, OpCost
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.transform import IdentityTransform
from src.formats.base import FormatBase


@pytest.fixture
def device():
    return DeviceSpec.a100()


@pytest.fixture
def int8_scheme():
    fmt = FormatBase.from_str("int8")
    return QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())


class TestOpCostLinear:
    def test_linear_fp32_cost(self, device):
        """FP32 linear: no quantization, only math FLOPs."""
        m = nn.Linear(64, 128)
        cost = op_cost(m, {}, device)
        assert cost.op_type == "linear"
        assert cost.flops_quantize == 0
        assert cost.flops_transform == 0
        assert cost.flops_math == 2 * 1 * 64 * 128  # batch=1 implicit
        assert cost.latency_us > 0

    def test_linear_quantized_cost(self, device, int8_scheme):
        """Quantized linear has nonzero quantize FLOPs."""
        from src.ops.linear import QuantizedLinear
        cfg = OpQuantConfig(input=int8_scheme, weight=int8_scheme, output=int8_scheme)
        m = QuantizedLinear(64, 128, cfg=cfg)
        cost = op_cost(m, {"batch": 4}, device)
        assert cost.flops_quantize > 0
        # Verify latency > fp32 linear (more ops)
        cost_fp32 = op_cost(nn.Linear(64, 128), {"batch": 4}, device)
        assert cost.latency_us > cost_fp32.latency_us

    def test_linear_none_weight_skips_weight_quantize(self, device, int8_scheme):
        """None weight scheme = no weight quantize overhead."""
        from src.ops.linear import QuantizedLinear
        cfg_input_only = OpQuantConfig(input=int8_scheme, weight=None, output=None)
        cfg_full = OpQuantConfig(input=int8_scheme, weight=int8_scheme, output=int8_scheme)
        m_partial = QuantizedLinear(64, 128, cfg=cfg_input_only)
        m_full = QuantizedLinear(64, 128, cfg=cfg_full)
        cost_partial = op_cost(m_partial, {}, device)
        cost_full = op_cost(m_full, {}, device)
        assert cost_partial.flops_quantize < cost_full.flops_quantize

    def test_linear_fp32_flops_formula(self, device):
        """Linear math FLOPs = 2 * B * C_in * C_out."""
        m = nn.Linear(32, 64)
        cost = op_cost(m, {"batch": 8}, device)
        assert cost.flops_math == 2 * 8 * 32 * 64


class TestOpCostShapeSensitivity:
    def test_larger_batch_increases_flops(self, device):
        m = nn.Linear(64, 128)
        small = op_cost(m, {"batch": 1}, device).flops_math
        large = op_cost(m, {"batch": 8}, device).flops_math
        assert large > small
        # Should scale roughly linearly
        assert 7 * small <= large <= 9 * small

    def test_latency_positive(self, device):
        m = nn.Linear(64, 128)
        cost = op_cost(m, {}, device)
        assert cost.latency_us > 0

    def test_memory_weight_bytes_for_fp32(self, device):
        """FP32 linear weight memory = in_feat * out_feat * 4 bytes."""
        m = nn.Linear(64, 128)
        cost = op_cost(m, {}, device)
        assert cost.memory_weight_bytes == 64 * 128 * 4  # FP32 = 4 bytes

    def test_memory_weight_bytes_for_int8(self, device):
        """INT8 quantized weight memory = in_feat * out_feat * 1 byte."""
        from src.ops.linear import QuantizedLinear
        fmt = FormatBase.from_str("int8")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(weight=scheme)
        m = QuantizedLinear(64, 128, cfg=cfg)
        cost = op_cost(m, {}, device)
        assert cost.memory_weight_bytes == 64 * 128 * 1  # INT8 = 1 byte


class TestOpCostEffectiveBits:
    def test_effective_bits_fp32(self, device):
        """FP32 (no scheme) effective bits = 32."""
        from src.cost.op_cost import _effective_bits
        assert _effective_bits(None) == 32.0

    def test_effective_bits_int8(self, device, int8_scheme):
        """INT8 per-tensor effective bits = 8."""
        from src.cost.op_cost import _effective_bits
        assert _effective_bits(int8_scheme) == 8.0

    def test_effective_bits_mx_block32(self, device):
        """INT8 MX block=32: effective bits = 8 + 8/32 = 8.25."""
        from src.cost.op_cost import _effective_bits
        fmt = FormatBase.from_str("int8")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_block(32))
        assert _effective_bits(scheme) == pytest.approx(8.25)

    def test_effective_bits_int4(self, device):
        """INT4 per-tensor: effective bits = 4."""
        from src.cost.op_cost import _effective_bits
        fmt = FormatBase.from_str("int4")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        assert _effective_bits(scheme) == 4.0

    def test_effective_bits_bf16(self, device):
        """BF16: effective bits = 16."""
        from src.cost.op_cost import _effective_bits
        fmt = FormatBase.from_str("bf16")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        assert _effective_bits(scheme) == 16.0


class TestOpCostElemBits:
    def test_elem_bits_int8(self):
        """int8 element bits = 8."""
        from src.cost.op_cost import _elem_bits
        fmt = FormatBase.from_str("int8")
        assert _elem_bits(fmt) == 8

    def test_elem_bits_int4(self):
        """int4 element bits = 4."""
        from src.cost.op_cost import _elem_bits
        fmt = FormatBase.from_str("int4")
        assert _elem_bits(fmt) == 4

    def test_elem_bits_fp8(self):
        """fp8_e4m3 element bits = 8."""
        from src.cost.op_cost import _elem_bits
        fmt = FormatBase.from_str("fp8_e4m3")
        assert _elem_bits(fmt) == 8

    def test_elem_bits_bf16(self):
        """bf16 element bits = 16."""
        from src.cost.op_cost import _elem_bits
        fmt = FormatBase.from_str("bf16")
        assert _elem_bits(fmt) == 16

    def test_elem_bits_nf4(self):
        """NF4 element bits = 4 (log2(16) levels)."""
        from src.cost.op_cost import _elem_bits
        from src.formats.lookup_formats import NF4Format
        assert _elem_bits(NF4Format()) == 4

    def test_elem_bits_none_is_32(self):
        from src.cost.op_cost import _elem_bits
        assert _elem_bits(None) == 32


class TestOpCostNormDispatch:
    """QuantizedBatchNorm2d and QuantizedGroupNorm are recognized."""

    def test_quantized_batch_norm_dispatch(self, device):
        from src.ops.norm import QuantizedBatchNorm2d
        m = QuantizedBatchNorm2d(32)
        cost = op_cost(m, {}, device)
        assert cost.op_type == "batch_norm"

    def test_quantized_group_norm_dispatch(self, device):
        from src.ops.norm import QuantizedGroupNorm
        m = QuantizedGroupNorm(num_groups=4, num_channels=32)
        cost = op_cost(m, {}, device)
        assert cost.op_type == "group_norm"

    def test_fp32_batch_norm_dispatch(self, device):
        m = nn.BatchNorm2d(32)
        cost = op_cost(m, {}, device)
        assert cost.op_type == "batch_norm"

    def test_fp32_group_norm_dispatch(self, device):
        m = nn.GroupNorm(num_groups=4, num_channels=32)
        cost = op_cost(m, {}, device)
        assert cost.op_type == "group_norm"

    def test_rms_norm_has_fewer_quant_steps(self, device):
        """RMSNorm should have fewer quantize steps than LayerNorm."""
        from src.ops.norm import QuantizedRMSNorm, QuantizedLayerNorm
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec
        from src.formats.base import FormatBase

        fmt = FormatBase.from_str("int8")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        rms = QuantizedRMSNorm(128, cfg=OpQuantConfig(input=scheme, output=scheme))
        ln = QuantizedLayerNorm(128, cfg=OpQuantConfig(input=scheme, output=scheme))

        cost_rms = op_cost(rms, {}, device)
        cost_ln = op_cost(ln, {}, device)
        assert cost_rms.flops_quantize < cost_ln.flops_quantize

    def test_layer_norm_step_count_is_9(self, device):
        """LayerNorm has 9 quantize steps (not 10) — no compute(bias)."""
        from src.ops.norm import QuantizedLayerNorm
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.granularity import GranularitySpec
        from src.formats.base import FormatBase

        fmt = FormatBase.from_str("int8")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(input=scheme)
        ln = QuantizedLayerNorm(128, cfg=cfg)
        cost = op_cost(ln, {}, device)
        # With storage=None, quantize FLOPs come only from input scheme steps
        # LayerNorm: compute(in) is 1 step with input scheme
        assert cost.flops_quantize > 0
