"""Per-operator cost functions.

Formulas: docs/refs/p6-cost-model-formulas.md §2-§4.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from .device import DeviceSpec
from .defaults import (
    QUANT_OPS_PER_ELEM_BASE, QUANT_OPS_PER_ELEM_MX,
    QUANT_OPS_PER_ELEM_BFLOAT, QUANT_OPS_PER_ELEM_LOOKUP,
    TRANSFORM_OPS_PER_ELEM_DEFAULT, TRANSFORM_OPS_PER_ELEM_HADAMARD,
    MX_SCALE_BITS,
)
from src.scheme.granularity import GranularityMode


@dataclass
class OpCost:
    op_name: str = ""
    op_type: str = ""
    flops_math: int = 0
    flops_quantize: int = 0
    flops_transform: int = 0
    bytes_read: int = 0
    bytes_write: int = 0
    latency_us: float = 0.0
    memory_weight_bytes: int = 0
    memory_activation_bytes: int = 0


# ── Helpers ────────────────────────────────────────────────────

def _elem_bits(fmt) -> int:
    """Return storage bits per element for a format.

    Convention (from src/formats/):
      IntFormat / LookupFormat: ebits=0, mbits = total bits
      FPFormat / BFloat16Format: ebits>0, mbits includes sign + implicit bit,
        so total = ebits + mbits - 1
    """
    if fmt is None:
        return 32
    if fmt.ebits == 0:
        return fmt.mbits
    return fmt.ebits + fmt.mbits - 1


def _effective_bits(scheme: Optional[object]) -> float:
    """Effective bits per element including per-block scale overhead (§4.1)."""
    if scheme is None:
        return 32.0
    fmt = scheme.format
    elem_bits = _elem_bits(fmt)
    if scheme.granularity.mode == GranularityMode.PER_BLOCK:
        return elem_bits + MX_SCALE_BITS / scheme.granularity.block_size
    return float(elem_bits)


def _quant_ops_per_elem(scheme) -> int:
    """Per-element quantization FLOPs for a scheme (§3.2)."""
    from src.formats.bf16_fp16 import BFloat16Format
    from src.formats.lookup_formats import LookupFormat
    fmt = scheme.format
    if isinstance(fmt, BFloat16Format):
        return QUANT_OPS_PER_ELEM_BFLOAT
    if isinstance(fmt, LookupFormat):
        return len(fmt.levels)
    if scheme.granularity.mode == GranularityMode.PER_BLOCK:
        return QUANT_OPS_PER_ELEM_MX
    return QUANT_OPS_PER_ELEM_BASE


def _granularity_flops(num_elem: int, scheme) -> int:
    """Additional FLOPs from granularity reduction (§3.2)."""
    mode = scheme.granularity.mode
    if mode == GranularityMode.PER_BLOCK:
        return num_elem * 2
    if mode in (GranularityMode.PER_TENSOR, GranularityMode.PER_CHANNEL):
        return num_elem
    return 0


def _transform_ops_per_elem(scheme) -> float:
    """Per-element transform FLOPs coefficient (§3.3)."""
    from src.scheme.transform import IdentityTransform
    t = scheme.transform
    if t is None or isinstance(t, IdentityTransform):
        return 0.0
    try:
        from src.transform.hadamard import HadamardTransform
        if isinstance(t, HadamardTransform):
            return float(TRANSFORM_OPS_PER_ELEM_HADAMARD)
    except ImportError:
        pass
    return float(TRANSFORM_OPS_PER_ELEM_DEFAULT)


def _quant_step_cost(num_elem: int, scheme: Optional[object]) -> tuple:
    """Compute (quant_flops, transform_flops, bytes_read, bytes_write) for one step (§3.2, §4.2)."""
    if scheme is None:
        return (0, 0, 0, 0)

    flops_q = num_elem * _quant_ops_per_elem(scheme) + _granularity_flops(num_elem, scheme)

    transform_ops = _transform_ops_per_elem(scheme)
    flops_t = int(num_elem * transform_ops) if transform_ops > 0 else 0

    # Read FP32 input, write quantized output (§4.2)
    eff_bits = _effective_bits(scheme)
    bytes_r = num_elem * 4
    bytes_w = int(num_elem * eff_bits / 8)

    return (flops_q, flops_t, bytes_r, bytes_w)


def _compute_latency(flops_math: int, flops_quantize: int, flops_transform: int,
                     bytes_read: int, bytes_write: int, device: DeviceSpec) -> float:
    """Roofline latency in microseconds (§2.1)."""
    total_flops = flops_math + flops_quantize + flops_transform
    total_bytes = bytes_read + bytes_write

    compute_time = total_flops / (device.peak_flops_fp32 * 1e12 * device.utilization)
    memory_time = total_bytes / (device.memory_bandwidth_gbs * 1e9 * device.utilization)

    return max(compute_time, memory_time) * device.kernel_overhead * 1e6


# ── Per-op implementations ─────────────────────────────────────

def _matmul_like_cost(m: nn.Module, shapes: dict, device: DeviceSpec, op_type: str) -> OpCost:
    """Cost for Linear / Matmul-like ops (§3.1)."""
    batch = shapes.get("batch", 1)
    in_features = shapes.get("in_features", getattr(m, "in_features", 64))
    out_features = shapes.get("out_features", getattr(m, "out_features", 128))

    # Math FLOPs: 2 × B × C_in × C_out
    flops_math = 2 * batch * in_features * out_features

    # Math memory (§4.3)
    weight_elem = in_features * out_features
    has_bias = getattr(m, "bias", None) is not None
    bias_elem = out_features if has_bias else 0
    input_elem = batch * in_features
    output_elem = batch * out_features

    bytes_r = (input_elem + weight_elem + bias_elem) * 4
    bytes_w = output_elem * 4

    # Quantize overhead
    cfg = getattr(m, "cfg", None)
    flops_q, flops_t, q_r, q_w = _linear_quant_overhead(
        cfg, input_elem, weight_elem, bias_elem, output_elem)
    bytes_r += q_r
    bytes_w += q_w

    # Memory footprint (§2.2)
    weight_scheme = cfg.weight if cfg else None
    input_scheme = cfg.input if cfg else None
    output_scheme = cfg.output if cfg else None

    mem_weight = int(weight_elem * _effective_bits(weight_scheme) / 8)
    mem_act = int(input_elem * _effective_bits(input_scheme) / 8
                  + output_elem * _effective_bits(output_scheme) / 8)

    latency = _compute_latency(flops_math, flops_q, flops_t, bytes_r, bytes_w, device)

    return OpCost(
        op_name="", op_type=op_type,
        flops_math=flops_math, flops_quantize=flops_q,
        flops_transform=flops_t,
        bytes_read=bytes_r, bytes_write=bytes_w,
        latency_us=latency,
        memory_weight_bytes=mem_weight, memory_activation_bytes=mem_act,
    )


def _linear_quant_overhead(cfg, input_elem, weight_elem, bias_elem, output_elem):
    """Quantize overhead for Linear forward (9 steps per §3.2)."""
    if cfg is None:
        return (0, 0, 0, 0)

    total_q = total_t = total_r = total_w = 0

    def _add(num_elem, scheme):
        nonlocal total_q, total_t, total_r, total_w
        q, t, r, w = _quant_step_cost(num_elem, scheme)
        total_q += q; total_t += t; total_r += r; total_w += w

    s = cfg.storage
    # storage steps: input, weight, bias
    _add(input_elem, s)
    _add(weight_elem, s)
    _add(bias_elem, s)
    # compute steps: input, weight
    _add(input_elem, cfg.input)
    _add(weight_elem, cfg.weight)
    # storage steps: out0, bias_add result, out1
    _add(output_elem, s)
    _add(output_elem, s)
    _add(output_elem, s)
    # compute output step
    _add(output_elem, cfg.output)

    return (total_q, total_t, total_r, total_w)


def _conv2d_cost(m: nn.Module, shapes: dict, device: DeviceSpec) -> OpCost:
    """Cost for Conv2d (§3.1)."""
    batch = shapes.get("batch", 1)
    h_in = shapes.get("height", 32)
    w_in = shapes.get("width", 32)

    kH = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
    kW = m.kernel_size[1] if isinstance(m.kernel_size, tuple) else m.kernel_size
    stride_h = m.stride[0] if isinstance(m.stride, tuple) else m.stride
    stride_w = m.stride[1] if isinstance(m.stride, tuple) else m.stride
    pad_h = m.padding[0] if isinstance(m.padding, tuple) else m.padding
    pad_w = m.padding[1] if isinstance(m.padding, tuple) else m.padding

    h_out = (h_in + 2 * pad_h - kH) // stride_h + 1
    w_out = (w_in + 2 * pad_w - kW) // stride_w + 1

    c_in = m.in_channels
    c_out = m.out_channels

    # Math FLOPs: 2 × B × C_out × H_out × W_out × C_in × kH × kW
    flops_math = 2 * batch * c_out * h_out * w_out * c_in * kH * kW

    # Memory
    weight_elem = c_out * c_in * kH * kW
    has_bias = getattr(m, "bias", None) is not None
    bias_elem = c_out if has_bias else 0
    input_elem = batch * c_in * h_in * w_in
    output_elem = batch * c_out * h_out * w_out

    bytes_r = (input_elem + weight_elem + bias_elem) * 4
    bytes_w = output_elem * 4

    cfg = getattr(m, "cfg", None)
    flops_q, flops_t, q_r, q_w = _linear_quant_overhead(
        cfg, input_elem, weight_elem, bias_elem, output_elem)
    bytes_r += q_r
    bytes_w += q_w

    weight_scheme = cfg.weight if cfg else None
    input_scheme = cfg.input if cfg else None
    output_scheme = cfg.output if cfg else None

    mem_weight = int(weight_elem * _effective_bits(weight_scheme) / 8)
    mem_act = int(input_elem * _effective_bits(input_scheme) / 8
                  + output_elem * _effective_bits(output_scheme) / 8)

    latency = _compute_latency(flops_math, flops_q, flops_t, bytes_r, bytes_w, device)

    return OpCost(
        op_name="", op_type="conv2d",
        flops_math=flops_math, flops_quantize=flops_q,
        flops_transform=flops_t,
        bytes_read=bytes_r, bytes_write=bytes_w,
        latency_us=latency,
        memory_weight_bytes=mem_weight, memory_activation_bytes=mem_act,
    )


def _norm_cost(m: nn.Module, shapes: dict, device: DeviceSpec, op_type: str) -> OpCost:
    """Cost for BatchNorm / LayerNorm / RMSNorm / GroupNorm (§3.1)."""
    batch = shapes.get("batch", 1)

    if hasattr(m, "num_features"):
        n = m.num_features  # BatchNorm
    elif hasattr(m, "normalized_shape"):
        n = m.normalized_shape[-1] if m.normalized_shape else shapes.get("n", 128)
    elif hasattr(m, "d_model"):
        n = m.d_model
    else:
        n = shapes.get("n", 128)

    ops_table = {
        "batch_norm": 4, "layer_norm": 4, "group_norm": 4, "rms_norm": 2,
    }
    flops_per_elem = ops_table.get(op_type, 4)
    flops_math = flops_per_elem * batch * n

    input_elem = batch * n
    output_elem = batch * n
    weight_elem = n
    bias_elem = n

    bytes_r = (input_elem + weight_elem + bias_elem) * 4
    bytes_w = output_elem * 4

    cfg = getattr(m, "cfg", None)
    flops_q = flops_t = q_r = q_w = 0

    if cfg is not None:
        steps = [
            (input_elem, cfg.storage),
            (weight_elem, cfg.storage),
            (bias_elem, cfg.storage),
            (input_elem, cfg.input),
            (weight_elem, cfg.weight if hasattr(cfg, "weight") else None),
            (input_elem, cfg.storage),   # normed intermediate
            (output_elem, cfg.storage),  # out0
            (output_elem, cfg.storage),  # out1
            (output_elem, cfg.output),
        ]
        if op_type == "rms_norm":
            steps = steps[:7]
        for elem, scheme in steps:
            q, t, r, w = _quant_step_cost(elem, scheme)
            flops_q += q; flops_t += t; q_r += r; q_w += w

    bytes_r += q_r
    bytes_w += q_w

    weight_scheme = cfg.weight if (cfg and hasattr(cfg, "weight")) else None
    input_scheme = cfg.input if cfg else None
    output_scheme = cfg.output if cfg else None

    mem_weight = int(weight_elem * _effective_bits(weight_scheme) / 8)
    mem_act = int(input_elem * _effective_bits(input_scheme) / 8
                  + output_elem * _effective_bits(output_scheme) / 8)

    latency = _compute_latency(flops_math, flops_q, flops_t, bytes_r, bytes_w, device)

    return OpCost(
        op_name="", op_type=op_type,
        flops_math=flops_math, flops_quantize=flops_q,
        flops_transform=flops_t,
        bytes_read=bytes_r, bytes_write=bytes_w,
        latency_us=latency,
        memory_weight_bytes=mem_weight, memory_activation_bytes=mem_act,
    )


def _activation_cost(m: nn.Module, shapes: dict, device: DeviceSpec, op_type: str) -> OpCost:
    """Cost for activation functions (§3.1)."""
    batch = shapes.get("batch", 1)
    n = shapes.get("n", shapes.get("in_features", 128))

    ops_table = {
        "gelu": 8, "silu": 4, "sigmoid": 4,
        "relu": 1, "relu6": 1, "leaky_relu": 1,
        "tanh": 5, "softmax": 3,
    }
    flops_math = ops_table.get(op_type, 1) * batch * n

    elem = batch * n
    bytes_r = elem * 4
    bytes_w = elem * 4

    cfg = getattr(m, "cfg", None)
    flops_q = flops_t = q_r = q_w = 0
    if cfg is not None:
        for e, scheme in [(elem, cfg.storage), (elem, cfg.input),
                          (elem, cfg.storage), (elem, cfg.output), (elem, cfg.storage)]:
            q, t, r, w = _quant_step_cost(e, scheme)
            flops_q += q; flops_t += t; q_r += r; q_w += w
    bytes_r += q_r
    bytes_w += q_w

    input_scheme = cfg.input if cfg else None
    output_scheme = cfg.output if cfg else None
    mem_act = int(elem * _effective_bits(input_scheme) / 8
                  + elem * _effective_bits(output_scheme) / 8)

    latency = _compute_latency(flops_math, flops_q, flops_t, bytes_r, bytes_w, device)

    return OpCost(
        op_name="", op_type=op_type,
        flops_math=flops_math, flops_quantize=flops_q,
        flops_transform=flops_t,
        bytes_read=bytes_r, bytes_write=bytes_w,
        latency_us=latency,
        memory_weight_bytes=0, memory_activation_bytes=mem_act,
    )


def _pool_cost(m: nn.Module, shapes: dict, device: DeviceSpec) -> OpCost:
    """Cost for pooling ops (§3.1)."""
    batch = shapes.get("batch", 1)
    c = shapes.get("channels", shapes.get("out_features", 64))
    h_in = shapes.get("height", 32)
    w_in = shapes.get("width", 32)

    flops_math = batch * c * h_in * w_in

    elem = batch * c * h_in * w_in
    bytes_r = elem * 4
    bytes_w = elem * 4

    cfg = getattr(m, "cfg", None)
    flops_q = flops_t = q_r = q_w = 0
    if cfg is not None:
        for e, scheme in [(elem, cfg.storage), (elem, cfg.input), (elem, cfg.output)]:
            q, t, r, w = _quant_step_cost(e, scheme)
            flops_q += q; flops_t += t; q_r += r; q_w += w
    bytes_r += q_r
    bytes_w += q_w

    input_scheme = cfg.input if cfg else None
    output_scheme = cfg.output if cfg else None
    mem_act = int(elem * _effective_bits(input_scheme) / 8
                  + elem * _effective_bits(output_scheme) / 8)

    latency = _compute_latency(flops_math, flops_q, flops_t, bytes_r, bytes_w, device)

    return OpCost(
        op_name="", op_type="adaptive_avg_pool",
        flops_math=flops_math, flops_quantize=flops_q,
        flops_transform=flops_t,
        bytes_read=bytes_r, bytes_write=bytes_w,
        latency_us=latency,
        memory_weight_bytes=0, memory_activation_bytes=mem_act,
    )


_KNOWN_LEAF_TYPES = None


def _get_known_leaf_types():
    global _KNOWN_LEAF_TYPES
    if _KNOWN_LEAF_TYPES is not None:
        return _KNOWN_LEAF_TYPES
    types = set()
    for mod_path, cls_name in [
        ("src.ops.linear", "QuantizedLinear"),
        ("src.ops.conv", "QuantizedConv2d"),
        ("src.ops.norm", "QuantizedLayerNorm"),
        ("src.ops.norm", "QuantizedRMSNorm"),
        ("src.ops.norm", "QuantizedBatchNorm2d"),
        ("src.ops.activations", "QuantizedActivation"),
        ("src.ops.softmax", "QuantizedSoftmax"),
        ("src.ops.pooling", "QuantizedAdaptiveAvgPool2d"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            types.add(getattr(mod, cls_name))
        except (ImportError, AttributeError):
            pass
    _KNOWN_LEAF_TYPES = tuple(types)
    return _KNOWN_LEAF_TYPES


def _dispatch_op_cost(m: nn.Module, shapes: dict, device: DeviceSpec) -> Optional[OpCost]:
    """Dispatch to the correct cost function for a given module."""
    from src.ops.linear import QuantizedLinear
    from src.ops.conv import QuantizedConv2d

    # Linear / QuantizedLinear
    if isinstance(m, (nn.Linear, QuantizedLinear)):
        return _matmul_like_cost(m, shapes, device, "linear")

    # Conv2d / QuantizedConv2d
    if isinstance(m, (nn.Conv2d, QuantizedConv2d)):
        return _conv2d_cost(m, shapes, device)

    # Norm
    try:
        from src.ops.norm import QuantizedLayerNorm, QuantizedRMSNorm
        if isinstance(m, (nn.LayerNorm, QuantizedLayerNorm)):
            return _norm_cost(m, shapes, device, "layer_norm")
        if isinstance(m, QuantizedRMSNorm):
            return _norm_cost(m, shapes, device, "rms_norm")
    except ImportError:
        pass

    if isinstance(m, nn.BatchNorm2d):
        return _norm_cost(m, shapes, device, "batch_norm")

    # Activation
    act_map = {
        nn.GELU: "gelu", nn.SiLU: "silu", nn.Sigmoid: "sigmoid",
        nn.ReLU: "relu", nn.ReLU6: "relu6", nn.LeakyReLU: "leaky_relu",
        nn.Tanh: "tanh", nn.Softmax: "softmax",
    }
    for cls, op_type in act_map.items():
        if isinstance(m, cls):
            return _activation_cost(m, shapes, device, op_type)

    # Pooling
    if isinstance(m, nn.AdaptiveAvgPool2d):
        return _pool_cost(m, shapes, device)

    # Quantized activation/softmax/pool from src.ops
    try:
        from src.ops.activations import QuantizedActivation
        if isinstance(m, QuantizedActivation):
            return _activation_cost(m, shapes, device, "activation")
    except ImportError:
        pass

    try:
        from src.ops.softmax import QuantizedSoftmax
        if isinstance(m, QuantizedSoftmax):
            return _activation_cost(m, shapes, device, "softmax")
    except ImportError:
        pass

    try:
        from src.ops.pooling import QuantizedAdaptiveAvgPool2d
        if isinstance(m, QuantizedAdaptiveAvgPool2d):
            return _pool_cost(m, shapes, device)
    except ImportError:
        pass

    return None


def op_cost(m: nn.Module, shapes: dict, device: DeviceSpec) -> OpCost:
    """Compute OpCost for a single nn.Module.

    Args:
        m: The module (quantized or FP32).
        shapes: Dict with optional keys: batch, in_features, out_features,
                height, width, n, channels. Missing keys use module attributes
                or safe defaults.
        device: DeviceSpec for the target hardware.

    Returns:
        OpCost with all fields populated. op_type="unknown" if unrecognized.
    """
    filled = dict(shapes)
    filled.setdefault("batch", 1)
    if hasattr(m, "in_features"):
        filled.setdefault("in_features", m.in_features)
    if hasattr(m, "out_features"):
        filled.setdefault("out_features", m.out_features)

    cost = _dispatch_op_cost(m, filled, device)
    if cost is None:
        return OpCost(op_name="", op_type="unknown")
    return cost
