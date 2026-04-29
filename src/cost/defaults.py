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
