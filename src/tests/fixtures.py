"""
Shared test fixtures: seeded tensor generators and representative mx_specs configs.
"""
import torch
import pytest
from mx.specs import MxSpecs, finalize_mx_specs


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
# Representative mx_specs configurations
# ---------------------------------------------------------------------------

def mxfp8_e4m3_specs(block_size=32, quantize_backprop=True):
    """MXFP8-E4M3 specs."""
    return finalize_mx_specs({
        "w_elem_format": "fp8_e4m3",
        "a_elem_format": "fp8_e4m3",
        "block_size": block_size,
        "bfloat": 16,
        "quantize_backprop": quantize_backprop,
    })


def mxfp8_e5m2_specs(block_size=32, quantize_backprop=True):
    """MXFP8-E5M2 specs."""
    return finalize_mx_specs({
        "w_elem_format": "fp8_e5m2",
        "a_elem_format": "fp8_e5m2",
        "block_size": block_size,
        "bfloat": 16,
        "quantize_backprop": quantize_backprop,
    })


def mxfp6_e3m2_specs(block_size=32, quantize_backprop=True):
    """MXFP6-E3M2 specs."""
    return finalize_mx_specs({
        "w_elem_format": "fp6_e3m2",
        "a_elem_format": "fp6_e3m2",
        "block_size": block_size,
        "bfloat": 16,
        "quantize_backprop": quantize_backprop,
    })


def mxfp4_e2m1_specs(block_size=32, quantize_backprop=True):
    """MXFP4-E2M1 specs."""
    return finalize_mx_specs({
        "w_elem_format": "fp4_e2m1",
        "a_elem_format": "fp4_e2m1",
        "block_size": block_size,
        "bfloat": 16,
        "quantize_backprop": quantize_backprop,
    })


def mxint8_specs(block_size=32, quantize_backprop=True):
    """MXINT8 specs."""
    return finalize_mx_specs({
        "w_elem_format": "int8",
        "a_elem_format": "int8",
        "block_size": block_size,
        "bfloat": 16,
        "quantize_backprop": quantize_backprop,
    })


def mxint4_specs(block_size=32, quantize_backprop=True):
    """MXINT4 specs."""
    return finalize_mx_specs({
        "w_elem_format": "int4",
        "a_elem_format": "int4",
        "block_size": block_size,
        "bfloat": 16,
        "quantize_backprop": quantize_backprop,
    })


def bfloat16_specs():
    """Bfloat16-only specs (no MX quantization)."""
    return finalize_mx_specs({"bfloat": 16})


def mixed_mxfp8_weight_fp4_act_specs(block_size=32):
    """Mixed format: FP8 weights, FP4 activations."""
    return finalize_mx_specs({
        "w_elem_format": "fp8_e4m3",
        "a_elem_format": "fp4_e2m1",
        "block_size": block_size,
        "bfloat": 16,
    })


def no_quant_specs():
    """No quantization specs (returns None like the old code)."""
    return None


# All representative MX specs for parametrized tests
ALL_MX_SPECS = [
    ("mxfp8_e4m3", mxfp8_e4m3_specs()),
    ("mxfp6_e3m2", mxfp6_e3m2_specs()),
    ("mxfp4_e2m1", mxfp4_e2m1_specs()),
    ("mxint8", mxint8_specs()),
    ("bfloat16", bfloat16_specs()),
]

# MX specs with backward pass quantization disabled
ALL_MX_SPECS_NO_BP = [
    ("mxfp8_e4m3_no_bp", mxfp8_e4m3_specs(quantize_backprop=False)),
    ("mxfp4_e2m1_no_bp", mxfp4_e2m1_specs(quantize_backprop=False)),
]
