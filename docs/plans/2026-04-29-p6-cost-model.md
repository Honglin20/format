# P6 Coarse Cost Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> **Formula authority:** `docs/refs/p6-cost-model-formulas.md` — all calculations must match exactly.
> **Architecture:** `docs/architecture/007-p6-cost-model.md`

**Goal:** Build a coarse cost model that estimates latency and memory for quantized models without real deployment.

**Architecture:** New `src/cost/` package with 6 modules (defaults, device, op_cost, model_cost, report, __init__). Integrate into `QuantSession.estimate_cost()` as a thin delegation layer. Pipeline `run_experiment()` auto-attaches cost data.

**Tech Stack:** Python, PyTorch (nn.Module introspection only, no forward pass), dataclasses, optional pandas.

---

### Task 1: Package skeleton — defaults.py + device.py + `__init__.py`

**Files:**
- Create: `src/cost/__init__.py`
- Create: `src/cost/defaults.py`
- Create: `src/cost/device.py`

**Step 1: Create `src/cost/defaults.py`**

Copy the constants from `docs/refs/p6-cost-model-formulas.md` §5 exactly:

```python
"""Tunable constants for the coarse cost model. Adjust for different hardware targets."""

# ── Device ─────────────────────────────────────────────────
DEFAULT_PEAK_FLOPS_FP32 = 19.5       # TFLOPS (A100)
DEFAULT_MEMORY_BANDWIDTH_GBS = 2039  # GB/s (A100 80GB)
DEFAULT_DEVICE_MEMORY_GB = 80.0

# ── Roofline correction ────────────────────────────────────
DEFAULT_UTILIZATION = 0.4
DEFAULT_KERNEL_OVERHEAD = 1.3

# ── Quantize per-element ops ───────────────────────────────
QUANT_OPS_PER_ELEM_BASE = 5
QUANT_OPS_PER_ELEM_MX = 8
QUANT_OPS_PER_ELEM_BFLOAT = 2
QUANT_OPS_PER_ELEM_LOOKUP = 16       # NF4 = 16 levels

# ── Transform per-element ops ──────────────────────────────
TRANSFORM_OPS_PER_ELEM_DEFAULT = 2
TRANSFORM_OPS_PER_ELEM_HADAMARD = 2

# ── MX constants ───────────────────────────────────────────
MX_SCALE_BITS = 8
```

**Step 2: Create `src/cost/device.py`**

```python
"""DeviceSpec: GPU capability profile for roofline modeling."""
from dataclasses import dataclass
from . import defaults


@dataclass
class DeviceSpec:
    peak_flops_fp32: float    # TFLOPS
    memory_bandwidth_gbs: float  # GB/s
    device_memory_gb: float
    utilization: float = 0.4
    kernel_overhead: float = 1.3

    @staticmethod
    def a100() -> "DeviceSpec":
        return DeviceSpec(
            peak_flops_fp32=defaults.DEFAULT_PEAK_FLOPS_FP32,
            memory_bandwidth_gbs=defaults.DEFAULT_MEMORY_BANDWIDTH_GBS,
            device_memory_gb=defaults.DEFAULT_DEVICE_MEMORY_GB,
            utilization=defaults.DEFAULT_UTILIZATION,
            kernel_overhead=defaults.DEFAULT_KERNEL_OVERHEAD,
        )
```

**Step 3: Create `src/cost/__init__.py`**

```python
"""Coarse cost model for quantized model latency and memory estimation."""
from .device import DeviceSpec
from .report import CostReport
from .model_cost import analyze_model_cost

__all__ = ["DeviceSpec", "CostReport", "analyze_model_cost"]
```

**Step 4: Commit**

```bash
git add src/cost/__init__.py src/cost/defaults.py src/cost/device.py
git commit -m "feat(cost): add package skeleton — defaults, DeviceSpec, __init__"
```

---

### Task 2: op_cost.py — per-operator cost functions

**Files:**
- Create: `src/cost/op_cost.py`
- Create: `src/tests/test_cost_op_cost.py`

**Step 1: Write failing test skeleton**

File: `src/tests/test_cost_op_cost.py`

```python
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

    def test_linear_quantized_cost(self, device):
        """Quantized linear has nonzero quantize + transform FLOPs."""
        from src.ops.linear import QuantizedLinear
        fmt = FormatBase.from_str("int8")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor(),
                             transform=IdentityTransform())
        cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
        m = QuantizedLinear(64, 128, cfg=cfg)
        cost = op_cost(m, {"batch": 4}, device)
        assert cost.flops_quantize > 0
        # input(4×64), weight(128×64), bias(128), output(4×128) — 9 quantize steps
        # Verify latency > fp32 linear
        cost_fp32 = op_cost(nn.Linear(64, 128), {"batch": 4}, device)
        assert cost.latency_us > cost_fp32.latency_us

    def test_linear_none_schemes_skip_quantize(self, device):
        """None scheme = no quantize overhead for that step."""
        from src.ops.linear import QuantizedLinear
        fmt = FormatBase.from_str("int8")
        scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
        cfg = OpQuantConfig(input=scheme, weight=None, output=None)
        m = QuantizedLinear(64, 128, cfg=cfg)
        cost = op_cost(m, {}, device)
        # Only 3 storage + 1 compute quantize steps (input only)
        full_cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
        m_full = QuantizedLinear(64, 128, cfg=full_cfg)
        cost_full = op_cost(m_full, {}, device)
        assert cost.flops_quantize < cost_full.flops_quantize


class TestOpCostShapeSensitivity:
    def test_larger_batch_increases_flops(self, device):
        m = nn.Linear(64, 128)
        small = op_cost(m, {"batch": 1}, device).flops_math
        large = op_cost(m, {"batch": 8}, device).flops_math
        assert large > small
        # Should scale roughly linearly
        assert 7 * small < large < 9 * small
```

Run: `pytest src/tests/test_cost_op_cost.py -v`
Expected: FAIL (module not found)

**Step 2: Implement `src/cost/op_cost.py`**

```python
"""Per-operator cost functions.

Each function computes OpCost for one operator type.
Formulas: docs/refs/p6-cost-model-formulas.md §3-§4.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from .device import DeviceSpec
from .defaults import (
    QUANT_OPS_PER_ELEM_BASE, QUANT_OPS_PER_ELEM_MX,
    QUANT_OPS_PER_ELEM_BFLOAT, QUANT_OPS_PER_ELEM_LOOKUP,
    TRANSFORM_OPS_PER_ELEM_DEFAULT, TRANSFORM_OPS_PER_ELEM_HADAMARD,
    MX_SCALE_BITS,
)
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularityMode
from src.scheme.transform import IdentityTransform
from src.formats.lookup_formats import LookupFormat
from src.formats.bfloat_formats import BFloat16Format


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
    """Return storage bits per element for a format."""
    if fmt is None:
        return 32
    return fmt.mbits + fmt.ebits + 1  # sign + exp + mantissa


def _effective_bits(scheme: Optional[QuantScheme]) -> float:
    """Effective bits per element including per-block scale overhead (§4.1)."""
    if scheme is None:
        return 32.0
    fmt = scheme.format
    elem_bits = _elem_bits(fmt)
    if scheme.granularity.mode == GranularityMode.PER_BLOCK:
        return elem_bits + MX_SCALE_BITS / scheme.granularity.block_size
    return float(elem_bits)


def _quant_ops_per_elem(scheme: QuantScheme) -> int:
    """Per-element quantization FLOPs for a scheme (§6.1)."""
    fmt = scheme.format
    if isinstance(fmt, BFloat16Format):
        return QUANT_OPS_PER_ELEM_BFLOAT
    if isinstance(fmt, LookupFormat):
        return len(fmt.levels)
    if scheme.granularity.mode == GranularityMode.PER_BLOCK:
        return QUANT_OPS_PER_ELEM_MX
    return QUANT_OPS_PER_ELEM_BASE


def _granularity_flops(num_elem: int, scheme: QuantScheme) -> int:
    """Additional FLOPs from granularity reduction (§6.2)."""
    mode = scheme.granularity.mode
    if mode == GranularityMode.PER_BLOCK:
        return num_elem * 2
    if mode in (GranularityMode.PER_TENSOR, GranularityMode.PER_CHANNEL):
        return num_elem
    return 0


def _transform_ops_per_elem(scheme: QuantScheme) -> float:
    """Per-element transform FLOPs (§7)."""
    from src.transform.hadamard import HadamardTransform
    t = scheme.transform
    if t is None or isinstance(t, IdentityTransform):
        return 0.0
    if isinstance(t, HadamardTransform):
        # Log2(N) will be multiplied by num_elem; we just return coefficient * log2
        # Caller provides num_elem and the padded dim separately
        return float(TRANSFORM_OPS_PER_ELEM_HADAMARD)
    return float(TRANSFORM_OPS_PER_ELEM_DEFAULT)


def _quant_step_cost(num_elem: int, scheme: Optional[QuantScheme]) -> tuple[int, int, int, int]:
    """Compute (quant_flops, transform_flops, bytes_read, bytes_write) for one step."""
    if scheme is None:
        return (0, 0, 0, 0)
    
    flops_q = num_elem * _quant_ops_per_elem(scheme) + _granularity_flops(num_elem, scheme)
    
    # Transform
    transform_ops = _transform_ops_per_elem(scheme)
    flops_t = int(num_elem * transform_ops) if transform_ops > 0 else 0
    
    # Memory: read FP32, write quantized
    eff_bits = _effective_bits(scheme)
    bytes_r = num_elem * 4
    bytes_w = int(num_elem * eff_bits / 8)
    
    return (flops_q, flops_t, bytes_r, bytes_w)


def _compute_latency(flops_math: int, flops_quantize: int, flops_transform: int,
                     bytes_read: int, bytes_write: int, device: DeviceSpec) -> float:
    """Roofline latency in microseconds (§3)."""
    total_flops = flops_math + flops_quantize + flops_transform
    total_bytes = bytes_read + bytes_write
    
    compute_time = total_flops / (device.peak_flops_fp32 * 1e12 * device.utilization)
    memory_time = total_bytes / (device.memory_bandwidth_gbs * 1e9 * device.utilization)
    
    return max(compute_time, memory_time) * device.kernel_overhead * 1e6


# ── Per-op cost functions ──────────────────────────────────────

# Quantize step counts per op type (§6.4 of formulas doc)
_QUANT_STEP_COUNTS = {
    "linear": 9,
    "conv1d": 9,
    "conv2d": 9,
    "conv3d": 9,
    "conv_transpose1d": 9,
    "conv_transpose2d": 9,
    "conv_transpose3d": 9,
    "batch_norm": 9,
    "layer_norm": 10,
    "rms_norm": 7,
    "group_norm": 10,
    "softmax": 5,
    "activation": 5,
    "adaptive_avg_pool": 3,
    "elemwise": 5,
    "matmul": 7,
    "bmm": 7,
}


def _dispatch_op_cost(m: nn.Module, shapes: dict, device: DeviceSpec) -> OpCost:
    """Detect op type from module class and dispatch to correct formula."""
    from src.ops.linear import QuantizedLinear
    from src.ops.conv import QuantizedConv2d
    from src.ops.norm import QuantizedLayerNorm, QuantizedRMSNorm
    
    # Detect type
    if isinstance(m, (nn.Linear, QuantizedLinear)):
        return _matmul_like_cost(m, shapes, device, "linear")
    if isinstance(m, (nn.Conv2d, QuantizedConv2d)):
        return _conv2d_cost(m, shapes, device)
    # ... more op types
    
    # Fallback: unknown op, return zero cost placeholder
    return OpCost(op_name="", op_type="unknown")


def _matmul_like_cost(m, shapes, device, op_type) -> OpCost:
    """Cost for Linear / Matmul / BMM (§3.1)."""
    batch = shapes.get("batch", 1)
    in_features = m.in_features if hasattr(m, "in_features") else m.weight.shape[1]
    out_features = m.out_features if hasattr(m, "out_features") else m.weight.shape[0]
    
    # Math FLOPs
    flops_math = 2 * batch * in_features * out_features
    
    # Quantize + transform: generic path using _QUANT_STEP_COUNTS
    cfg = getattr(m, "cfg", None)
    flops_quantize, flops_transform, bytes_r, bytes_w = _compute_quantize_overhead(
        cfg, shapes, op_type, device)
    
    # Math memory
    weight_elem = in_features * out_features
    bias_elem = out_features
    input_elem = batch * in_features
    output_elem = batch * out_features
    
    bytes_r += (input_elem + weight_elem + bias_elem) * 4
    bytes_w += output_elem * 4
    
    # Memory footprint
    weight_cfg = cfg.weight if cfg else None
    input_cfg = cfg.input if cfg else None
    output_cfg = cfg.output if cfg else None
    
    mem_weight = int(weight_elem * _effective_bits(weight_cfg) / 8)
    mem_act = int(input_elem * _effective_bits(input_cfg) / 8
                  + output_elem * _effective_bits(output_cfg) / 8)
    
    latency = _compute_latency(flops_math, flops_quantize, flops_transform,
                                bytes_r, bytes_w, device)
    
    return OpCost(
        op_name="", op_type=op_type,
        flops_math=flops_math, flops_quantize=flops_quantize,
        flops_transform=flops_transform,
        bytes_read=bytes_r, bytes_write=bytes_w,
        latency_us=latency,
        memory_weight_bytes=mem_weight, memory_activation_bytes=mem_act,
    )


def _compute_quantize_overhead(cfg, shapes, op_type, device) -> tuple[int, int, int, int]:
    """Compute total quantize + transform FLOPs and bytes for all steps."""
    if cfg is None:
        return (0, 0, 0, 0)
    
    num_steps = _QUANT_STEP_COUNTS.get(op_type, 0)
    # Estimate per-step element count from shapes
    batch = shapes.get("batch", 1)
    # Use a representative element count per step
    # For simplicity, use the largest tensor in the op
    total_elem = _estimate_total_quantized_elements(cfg, shapes, op_type)
    per_step_elem = total_elem // max(num_steps, 1)
    
    total_q, total_t, total_r, total_w = 0, 0, 0, 0
    
    # Build list of schemes from cfg for this op type
    schemes = _get_active_schemes(cfg, op_type)
    
    for scheme in schemes:
        if scheme is not None:
            q, t, r, w = _quant_step_cost(per_step_elem, scheme)
            total_q += q
            total_t += t
            total_r += r
            total_w += w
    
    return (total_q, total_t, total_r, total_w)


def _get_active_schemes(cfg, op_type) -> list:
    """Get list of QuantScheme from cfg for all quantization steps."""
    if cfg is None:
        return []
    
    # Build step list matching the forward pass pattern
    # storage(in, w, b, out0, bias_add, out1) + compute(in, w, out) = 9 for linear
    schemes = []
    storage = cfg.storage
    schemes.extend([storage] * 3)     # in, w, b storage
    schemes.append(cfg.input)          # input compute
    schemes.append(cfg.weight)         # weight compute
    schemes.extend([storage] * 2)     # out0, bias_add storage
    schemes.append(storage)            # out1 storage (if applicable)
    schemes.append(cfg.output)         # output compute
    return schemes


def _estimate_total_quantized_elements(cfg, shapes, op_type) -> int:
    """Estimate total elements across all quantized tensors for this op."""
    # Placeholder — will be refined per op type
    batch = shapes.get("batch", 1)
    in_feat = shapes.get("in_features", 64)
    out_feat = shapes.get("out_features", 128)
    return batch * (in_feat + out_feat) * 2 + in_feat * out_feat


def _conv2d_cost(m, shapes, device) -> OpCost:
    """Conv2d cost (§3.1)."""
    # Similar to _matmul_like_cost but with conv FLOPs formula
    batch = shapes.get("batch", 1)
    # ... (implemented in detail during task execution)
    return OpCost(op_type="conv2d")


def op_cost(m: nn.Module, shapes: dict, device: DeviceSpec) -> OpCost:
    """Compute OpCost for a single nn.Module.
    
    Args:
        m: The module (quantized or FP32).
        shapes: Dict with keys like "batch", "in_features", "out_features",
                "height", "width", etc. If a key is missing, reasonable
                defaults from module attributes are used.
        device: DeviceSpec for the target hardware.
    
    Returns:
        OpCost with all fields populated.
    """
    # Fill in missing shapes from module attributes
    filled = dict(shapes)
    if "batch" not in filled:
        filled["batch"] = 1
    if hasattr(m, "in_features"):
        filled.setdefault("in_features", m.in_features)
    if hasattr(m, "out_features"):
        filled.setdefault("out_features", m.out_features)
    
    return _dispatch_op_cost(m, filled, device)
```

Run: `pytest src/tests/test_cost_op_cost.py -v` — expect FAIL (need to iterate on implementation).

**Step 3: Iterate implementation until all tests pass**

Key: the test will fail initially because `_dispatch_op_cost` only handles Linear/Conv2d. Expand dispatch to cover all op types in Task 2.5.

**Step 4: Run tests and verify**

```bash
pytest src/tests/test_cost_op_cost.py -v
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add src/cost/op_cost.py src/tests/test_cost_op_cost.py
git commit -m "feat(cost): add op_cost.py — per-operator cost functions with roofline latency"
```

---

### Task 3: model_cost.py — model traversal and aggregation

**Files:**
- Create: `src/cost/model_cost.py`
- Create: `src/tests/test_cost_model_cost.py`

**Step 1: Write failing test**

```python
"""Tests for model-level cost analysis."""
import pytest
import torch
import torch.nn as nn
from src.cost.device import DeviceSpec
from src.cost.model_cost import analyze_model_cost
from src.cost.report import CostReport


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_analyze_fp32_model():
    model = TinyModel()
    report = analyze_model_cost(model, shapes={"batch": 4})
    
    assert isinstance(report, CostReport)
    assert len(report.layers) >= 2  # at least the two Linear layers
    assert report.total_latency_us > 0
    assert report.total_memory_bytes > 0
    
    # FP32 model has no quantize FLOPs
    for layer in report.layers:
        assert layer.flops_quantize == 0


def test_model_latency_is_sum_of_layers():
    model = TinyModel()
    report = analyze_model_cost(model, shapes={"batch": 1})
    layer_sum = sum(l.latency_us for l in report.layers)
    assert abs(report.total_latency_us - layer_sum) < 1.0


def test_memory_is_weight_sum_plus_max_activation():
    model = TinyModel()
    report = analyze_model_cost(model, shapes={"batch": 1})
    weight_sum = sum(l.memory_weight_bytes for l in report.layers)
    max_act = max(l.memory_activation_bytes for l in report.layers)
    assert report.total_memory_bytes == weight_sum + max_act
```

Run: `pytest src/tests/test_cost_model_cost.py -v`
Expected: FAIL

**Step 2: Implement `src/cost/model_cost.py`**

```python
"""Model-level cost analysis: walk nn.Module tree and aggregate per-layer costs."""
from __future__ import annotations
from typing import Optional

import torch.nn as nn

from .device import DeviceSpec
from .op_cost import op_cost, OpCost
from .report import CostReport


def analyze_model_cost(
    model: nn.Module,
    shapes: Optional[dict] = None,
    device: Optional[DeviceSpec] = None,
    model_name: str = "",
) -> CostReport:
    """Walk model and compute per-layer + total cost.
    
    Args:
        model: PyTorch model (quantized or FP32).
        shapes: Dict with shape hints (batch, in_features, etc.).
                Missing keys are inferred from module attributes.
        device: Target DeviceSpec. Default: DeviceSpec.a100().
        model_name: Label for the report.
    
    Returns:
        CostReport with per-layer OpCost entries and aggregate totals.
    """
    if shapes is None:
        shapes = {}
    if device is None:
        device = DeviceSpec.a100()
    
    layers = []
    for name, module in model.named_modules():
        cost = _try_cost(module, name, shapes, device)
        if cost is not None:
            layers.append(cost)
    
    return CostReport(layers=layers, model_name=model_name)


def _try_cost(module, name, shapes, device) -> OpCost | None:
    """Return OpCost for module if it's a recognized op type, else None."""
    # Skip container modules (no parameters, no compute)
    if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
        return None
    # Skip the root model itself
    if name == "":
        return None
    # Try to compute cost
    try:
        cost = op_cost(module, shapes, device)
        cost.op_name = name
        return cost
    except (TypeError, AttributeError):
        return None
```

**Step 3: Run tests and fix**

```bash
pytest src/tests/test_cost_model_cost.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/cost/model_cost.py src/tests/test_cost_model_cost.py
git commit -m "feat(cost): add model_cost.py — model traversal and layer aggregation"
```

---

### Task 4: report.py — CostReport output

**Files:**
- Create: `src/cost/report.py`
- Create: `src/tests/test_cost_report.py`

**Step 1: Write failing test**

```python
"""Tests for CostReport."""
import pytest
from src.cost.report import CostReport
from src.cost.op_cost import OpCost


def make_linear_cost(name="fc1", latency=10.0, mem_w=1000, mem_a=500):
    return OpCost(
        op_name=name, op_type="linear",
        flops_math=1000, latency_us=latency,
        memory_weight_bytes=mem_w, memory_activation_bytes=mem_a,
    )


def test_report_aggregates_latency():
    layers = [make_linear_cost("fc1", 10.0), make_linear_cost("fc2", 20.0)]
    report = CostReport(layers=layers)
    assert report.total_latency_us == 30.0


def test_report_aggregates_memory():
    layers = [
        make_linear_cost("fc1", mem_w=1000, mem_a=500),
        make_linear_cost("fc2", mem_w=2000, mem_a=800),
    ]
    report = CostReport(layers=layers)
    # weight sum + max activation
    assert report.total_memory_bytes == (1000 + 2000) + max(500, 800)


def test_report_to_dataframe():
    layers = [make_linear_cost("fc1"), make_linear_cost("fc2")]
    report = CostReport(layers=layers, model_name="test")
    df = report.to_dataframe()
    # Should return list of dicts (no pandas dependency)
    assert isinstance(df, list)
    assert len(df) == 2
    assert df[0]["op_name"] == "fc1"


def test_report_print_summary_runs():
    layers = [make_linear_cost("fc1")]
    report = CostReport(layers=layers, model_name="test")
    report.print_summary()  # should not raise


def test_print_comparison_runs():
    fp32 = CostReport([make_linear_cost("fc1", latency=5.0)], model_name="FP32")
    quant = CostReport([make_linear_cost("fc1", latency=8.0)], model_name="INT8")
    quant.print_comparison(fp32)  # should not raise
```

Run: `pytest src/tests/test_cost_report.py -v`
Expected: FAIL

**Step 2: Implement `src/cost/report.py`**

```python
"""CostReport: aggregate and format per-layer cost data."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .op_cost import OpCost


class CostReport:
    def __init__(self, layers: list[OpCost], model_name: str = ""):
        self.layers = list(layers)
        self.model_name = model_name
    
    @property
    def total_latency_us(self) -> float:
        return sum(l.latency_us for l in self.layers)
    
    @property
    def total_memory_bytes(self) -> int:
        weight_sum = sum(l.memory_weight_bytes for l in self.layers)
        max_act = max((l.memory_activation_bytes for l in self.layers), default=0)
        return weight_sum + max_act
    
    def summary(self) -> dict:
        return {
            "model": self.model_name or "model",
            "num_layers": len(self.layers),
            "total_latency_us": self.total_latency_us,
            "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
            "total_flops_math": sum(l.flops_math for l in self.layers),
            "total_flops_quantize": sum(l.flops_quantize for l in self.layers),
            "total_flops_transform": sum(l.flops_transform for l in self.layers),
        }
    
    def to_dataframe(self):
        rows = []
        for l in self.layers:
            rows.append({
                "op_name": l.op_name, "op_type": l.op_type,
                "flops_math": l.flops_math,
                "flops_quantize": l.flops_quantize,
                "flops_transform": l.flops_transform,
                "latency_us": round(l.latency_us, 2),
                "mem_weight_kb": round(l.memory_weight_bytes / 1024, 2),
                "mem_act_kb": round(l.memory_activation_bytes / 1024, 2),
            })
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows
    
    def print_summary(self):
        s = self.summary()
        print(f"=== Cost Report: {s['model']} ===")
        print(f"  Layers: {s['num_layers']}")
        print(f"  Total latency: {s['total_latency_us']:.1f} us")
        print(f"  Total memory:  {s['total_memory_mb']:.2f} MB")
        print(f"  Math FLOPs:    {s['total_flops_math']:,}")
        print(f"  Quant FLOPs:   {s['total_flops_quantize']:,}")
        print(f"  Transform FLOPs: {s['total_flops_transform']:,}")
    
    def print_per_layer(self):
        print(f"{'Layer':<24} {'Type':<16} {'Lat(us)':>10} {'Mem(kB)':>10}")
        print("-" * 62)
        for l in self.layers:
            mem_total = l.memory_weight_bytes + l.memory_activation_bytes
            print(f"  {l.op_name:<22} {l.op_type:<16} {l.latency_us:>10.2f} {mem_total/1024:>10.1f}")
    
    def print_comparison(self, baseline: "CostReport"):
        print(f"=== Cost Comparison: {self.model_name} vs {baseline.model_name} ===")
        lat_ratio = self.total_latency_us / max(baseline.total_latency_us, 1e-9)
        mem_ratio = self.total_memory_bytes / max(baseline.total_memory_bytes, 1)
        print(f"  Latency: {self.total_latency_us:.1f} / {baseline.total_latency_us:.1f} us "
              f"({lat_ratio:.2f}x)")
        print(f"  Memory:  {self.total_memory_bytes/1e6:.2f} / {baseline.total_memory_bytes/1e6:.2f} MB "
              f"({mem_ratio:.2f}x)")
```

**Step 3: Run tests**

```bash
pytest src/tests/test_cost_report.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/cost/report.py src/tests/test_cost_report.py
git commit -m "feat(cost): add CostReport — aggregation, formatting, and comparison"
```

---

### Task 5: Session integration — QuantSession.estimate_cost()

**Files:**
- Modify: `src/session.py` (add method at end of class)
- Modify: `src/tests/test_session.py` (or create integration test)

**Step 1: Add `estimate_cost()` to QuantSession**

Insert after `export_onnx()` (line ~248 in session.py):

```python
    # ------------------------------------------------------------------
    # Cost estimation (P6)
    # ------------------------------------------------------------------

    def estimate_cost(self, fp32: bool = False) -> "CostReport":
        """Estimate latency and memory for the current model.

        Unlike :meth:`analyze`, this does not require a forward pass —
        it inspects the model graph structure and quantization configs.

        Args:
            fp32: If True, estimate the fp32 baseline; otherwise the
                quantized model.

        Returns:
            CostReport with per-layer and total estimates.
        """
        from src.cost.model_cost import analyze_model_cost

        model = self.fp32_model if fp32 else self.qmodel
        if model is None:
            raise RuntimeError(
                "fp32_model not available (keep_fp32=False). "
                "Cannot estimate fp32 cost."
            )
        return analyze_model_cost(model)
```

**Step 2: Write integration test**

In `src/tests/test_cost_integration.py`:

```python
"""Integration tests for Cost model with QuantSession."""
import torch
import torch.nn as nn
from src.session import QuantSession
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.formats.base import FormatBase


def test_session_estimate_cost_fp32():
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
    
    session = QuantSession(model, cfg, keep_fp32=True)
    
    cost_q = session.estimate_cost()
    cost_fp32 = session.estimate_cost(fp32=True)
    
    assert cost_q.total_latency_us > 0
    assert cost_fp32.total_latency_us > 0
    # Quantized model should have lower memory
    assert cost_q.total_memory_bytes < cost_fp32.total_memory_bytes


def test_session_estimate_cost_no_fp32_raises():
    model = nn.Linear(64, 10)
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
    
    session = QuantSession(model, cfg, keep_fp32=False)
    
    import pytest
    with pytest.raises(RuntimeError, match="fp32_model"):
        session.estimate_cost(fp32=True)
```

**Step 3: Run tests**

```bash
pytest src/tests/test_cost_integration.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add src/session.py src/tests/test_cost_integration.py
git commit -m "feat(cost): integrate estimate_cost() into QuantSession"
```

---

### Task 6: Pipeline integration — run_experiment cost keys

**Files:**
- Modify: `src/pipeline/format_study.py` (run_experiment function)
- Modify: `src/pipeline/runner.py` (if needed)

**Step 1: Add cost keys to run_experiment return dict**

In `src/pipeline/format_study.py`, in `run_experiment()`, after the `session.compare()` call, add:

```python
    # Cost estimation (P6)
    cost = session.estimate_cost()
    cost_fp32 = session.estimate_cost(fp32=True)

    return {
        "accuracy": result["quant"],
        "fp32_accuracy": result["fp32"],
        "delta": result["delta"],
        "report": report,
        "session": session,
        "qsnr_per_layer": extract_metric_per_layer(report, "qsnr_db"),
        "mse_per_layer": extract_metric_per_layer(report, "mse"),
        "cost": cost,
        "cost_fp32": cost_fp32,
    }
```

**Step 2: Verify existing pipeline tests still pass**

```bash
pytest src/tests/test_pipeline_config.py -v
```

Expected: PASS (no regression)

**Step 3: Commit**

```bash
git add src/pipeline/format_study.py
git commit -m "feat(cost): attach cost estimates to run_experiment results"
```

---

### Task 7: Full test suite verification + docs update

**Files:**
- Modify: `docs/status/CURRENT.md`
- Verify: all tests pass

**Step 1: Run full test suite**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```

Expected: 1,348 + new cost tests, all pass.

**Step 2: Update CURRENT.md**

Mark P6 as complete, update next steps to P7.

**Step 3: Commit**

```bash
git add docs/status/CURRENT.md
git commit -m "docs: mark P6 Coarse Model complete, update status"
```

---

### Total: 7 tasks, 7 commits

| Task | What | Estimated effort |
|---|---|---|
| 1 | Package skeleton (defaults + device + `__init__`) | 15 min |
| 2 | op_cost.py core + tests | 45 min |
| 3 | model_cost.py traversal + tests | 20 min |
| 4 | report.py formatting + tests | 15 min |
| 5 | Session integration + test | 15 min |
| 6 | Pipeline integration | 5 min |
| 7 | Full suite verification + docs update | 10 min |
