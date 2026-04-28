"""
Shared test fixtures: seeded tensor generators and representative QuantScheme configs.
"""
import torch
import pytest

from src.scheme.quant_scheme import QuantScheme


# ---------------------------------------------------------------------------
# Seeded tensor generators
# ---------------------------------------------------------------------------

def make_tensor(shape, dtype=torch.float32, seed=42, device="cpu"):
    """Create a deterministic random tensor."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device=device, generator=gen)


def make_weight(in_features, out_features, dtype=torch.float32, seed=43, device="cpu"):
    """Create a deterministic weight matrix."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(out_features, in_features, dtype=dtype, device=device, generator=gen)


def make_bias(size, dtype=torch.float32, seed=44, device="cpu"):
    """Create a deterministic bias vector."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(size, dtype=dtype, device=device, generator=gen)


# ---------------------------------------------------------------------------
# Representative QuantScheme configurations
# ---------------------------------------------------------------------------

def mxfp8_e4m3_scheme(block_size=32):
    """MXFP8-E4M3 scheme."""
    return QuantScheme.mxfp("fp8_e4m3", block_size=block_size)


def mxfp8_e5m2_scheme(block_size=32):
    """MXFP8-E5M2 scheme."""
    return QuantScheme.mxfp("fp8_e5m2", block_size=block_size)


def mxfp6_e3m2_scheme(block_size=32):
    """MXFP6-E3M2 scheme."""
    return QuantScheme.mxfp("fp6_e3m2", block_size=block_size)


def mxfp4_e2m1_scheme(block_size=32):
    """MXFP4-E2M1 scheme."""
    return QuantScheme.mxfp("fp4_e2m1", block_size=block_size)


def mxint8_scheme(block_size=32):
    """MXINT8 scheme."""
    return QuantScheme.mxfp("int8", block_size=block_size)


def mxint4_scheme(block_size=32):
    """MXINT4 scheme."""
    return QuantScheme.mxfp("int4", block_size=block_size)


def mxint2_scheme(block_size=32):
    """MXINT2 scheme."""
    return QuantScheme.mxfp("int2", block_size=block_size)


def mxfp6_e2m3_scheme(block_size=32):
    """MXFP6-E2M3 scheme."""
    return QuantScheme.mxfp("fp6_e2m3", block_size=block_size)


def float16_scheme():
    """Float16-only scheme (no MX quantization)."""
    return QuantScheme.per_tensor("float16")


def bfloat16_scheme():
    """Bfloat16-only scheme (no MX quantization)."""
    return QuantScheme.per_tensor("bfloat16")


def mixed_mxfp8_weight_fp4_act_scheme(block_size=32):
    """Mixed format: FP8 weights, FP4 activations."""
    return QuantScheme.mxfp("fp8_e4m3", block_size=block_size)


# All representative schemes for parametrized tests
ALL_MX_SCHEMES = [
    ("mxfp8_e5m2", mxfp8_e5m2_scheme()),
    ("mxfp8_e4m3", mxfp8_e4m3_scheme()),
    ("mxfp6_e3m2", mxfp6_e3m2_scheme()),
    ("mxfp6_e2m3", mxfp6_e2m3_scheme()),
    ("mxfp4_e2m1", mxfp4_e2m1_scheme()),
    ("mxint8", mxint8_scheme()),
    ("mxint4", mxint4_scheme()),
    ("mxint2", mxint2_scheme()),
    ("float16", float16_scheme()),
    ("bfloat16", bfloat16_scheme()),
    ("mixed_fp8w_fp4a", mixed_mxfp8_weight_fp4_act_scheme()),
]
