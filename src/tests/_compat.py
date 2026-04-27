"""
Internal compat helpers for tests only.

Provides scheme_from_mx_specs() to convert golden-reference mx_specs dicts
into QuantScheme objects, and op_config_from_mx_specs() to convert them into
OpQuantConfig for operator equivalence tests.

This is the ONLY place in src/ that should reference mx_specs-style dicts —
all other test code uses QuantScheme / OpQuantConfig.
"""
from dataclasses import dataclass
from typing import Optional

from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.formats.base import FormatBase


@dataclass
class SchemeWithCompatInfo:
    """A QuantScheme plus extra info from mx_specs that isn't captured in the scheme.

    New APIs (quantize, quantize_bfloat) take these as separate function args.
    """
    scheme: QuantScheme
    allow_denorm: bool = True
    quantize_backprop: bool = True


def scheme_from_mx_specs(mx_specs, use_mx_format=False):
    """Convert an mx_specs dict (from golden .pt files) to a SchemeWithCompatInfo.

    Returns None if mx_specs represents no active quantization.

    Args:
        mx_specs: Dict from golden .pt file.
        use_mx_format: If True, prefer w_elem_format/a_elem_format over bfloat.
    """
    if mx_specs is None:
        return None

    fmt = _format_from_mx_specs_dict(mx_specs, use_mx_format=use_mx_format)
    if fmt is None:
        return None

    block_size = mx_specs.get("block_size", 0)
    if block_size > 0:
        granularity = GranularitySpec.per_block(block_size)
    else:
        granularity = GranularitySpec.per_tensor()

    round_mode = mx_specs.get("round", "nearest")

    scheme = QuantScheme(format=fmt, granularity=granularity,
                         round_mode=round_mode)

    return SchemeWithCompatInfo(
        scheme=scheme,
        allow_denorm=mx_specs.get("bfloat_subnorms", True),
        quantize_backprop=mx_specs.get("quantize_backprop", True),
    )


def _format_from_mx_specs_dict(mx_specs, use_mx_format=False):
    """Derive a FormatBase from an mx_specs dict."""
    if use_mx_format:
        for key in ("w_elem_format", "a_elem_format"):
            fmt_name = mx_specs.get(key)
            if fmt_name and isinstance(fmt_name, str):
                return FormatBase.from_str(fmt_name)

    bfloat = mx_specs.get("bfloat", 0)
    fp = mx_specs.get("fp", 0)

    if bfloat > 0 and fp > 0:
        raise ValueError("Cannot set both [bfloat] and [fp] in mx_specs.")
    elif bfloat > 9:
        if bfloat == 16:
            from src.formats.bf16_fp16 import BFloat16Format
            return BFloat16Format()
        from src.formats.fp_formats import FPFormat
        mbits = bfloat - 7
        return FPFormat(name=f"bfloat{bfloat}", ebits=8, mbits=mbits)
    elif bfloat > 0 and bfloat <= 9:
        raise ValueError("Cannot set [bfloat] <= 9 in mx_specs.")
    elif fp > 6:
        from src.formats.fp_formats import FPFormat
        mantissa_bits = fp - 6
        mbits = mantissa_bits + 2
        return FPFormat(name=f"fp{fp}", ebits=5, mbits=mbits)
    elif fp > 0 and fp <= 6:
        raise ValueError("Cannot set [fp] <= 6 in mx_specs.")

    if not use_mx_format:
        for key in ("w_elem_format", "a_elem_format"):
            fmt_name = mx_specs.get(key)
            if fmt_name and isinstance(fmt_name, str):
                return FormatBase.from_str(fmt_name)

    return None


# ---------------------------------------------------------------------------
# OpQuantConfig adapter (operator-level) — two-level model
# ---------------------------------------------------------------------------

def _elem_scheme(mx_specs: dict, round_key: str = "round_output") -> Optional[QuantScheme]:
    """Build an elemwise QuantScheme from mx_specs bfloat/fp keys, or None."""
    bfloat = mx_specs.get("bfloat", 0)
    fp = mx_specs.get("fp", 0)
    if bfloat == 0 and fp == 0:
        return None

    fmt = _format_from_mx_specs_dict(mx_specs, use_mx_format=False)
    if fmt is None:
        return None

    round_mode = mx_specs.get(round_key, mx_specs.get("round", "nearest"))
    return QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor(),
                       round_mode=round_mode)


def _mx_scheme(mx_specs: dict, fmt_key: str, block_size: int,
               round_key: str, block_axis: int = -1) -> Optional[QuantScheme]:
    """Build an MX block QuantScheme from mx_specs, or None if format is unset."""
    fmt_name = mx_specs.get(fmt_key)
    if fmt_name is None:
        return None
    fmt = FormatBase.from_str(fmt_name)
    bs = block_size if block_size > 0 else 32
    round_mode = mx_specs.get(round_key, mx_specs.get("round", "nearest"))
    return QuantScheme(format=fmt, granularity=GranularitySpec.per_block(bs, axis=block_axis),
                       round_mode=round_mode)


def op_config_from_mx_specs(mx_specs: dict, op_type: str = "linear") -> OpQuantConfig:
    """Convert an mx_specs dict to an OpQuantConfig for equivalence tests.

    Two-level model: storage = elemwise (uniform), compute = MX (per-role).

    Args:
        mx_specs: Dict from golden .pt file or mx_specs default dict.
        op_type: "linear", "matmul", "conv", or "conv_transpose".

    Returns:
        OpQuantConfig with storage + per-role compute fields.
    """
    block_size = mx_specs.get("block_size", 0)
    quantize_backprop = mx_specs.get("quantize_backprop", True)

    # Storage: elemwise scheme (uniform across all tensors)
    storage = _elem_scheme(mx_specs, "round_output")

    # --- Forward compute schemes ---
    input_mx_axis = 1 if op_type in ("conv", "conv_transpose") else -1
    input_mx = _mx_scheme(mx_specs, "a_elem_format", block_size, "round_mx_output",
                           block_axis=input_mx_axis)

    if op_type == "conv":
        weight_mx_axis = 1
    elif op_type == "conv_transpose":
        weight_mx_axis = 0
    elif op_type == "matmul":
        weight_mx_axis = -2
    else:
        weight_mx_axis = -1
    weight_mx = _mx_scheme(mx_specs, "w_elem_format", block_size, "round_mx_output",
                            block_axis=weight_mx_axis)

    if not quantize_backprop:
        return OpQuantConfig(
            storage=storage, input=input_mx, weight=weight_mx,
        )

    # --- Backward compute schemes ---
    go_elem = _elem_scheme(mx_specs, "round_grad_input")

    if op_type == "linear":
        return _linear_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem)
    elif op_type == "matmul":
        return _matmul_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem)
    elif op_type == "conv":
        return _conv_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem)
    elif op_type == "conv_transpose":
        return _conv_transpose_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")


def _linear_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem):
    """Build OpQuantConfig with backward fields for linear."""
    a_fmt_bp = mx_specs.get("a_elem_format_bp")
    a_fmt_bp_ex = mx_specs.get("a_elem_format_bp_ex") or a_fmt_bp
    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_ex}, "a_elem_format",
                              block_size, "round_mx_input_grad_weight", block_axis=-2) if a_fmt_bp_ex else None
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_ex}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_weight", block_axis=-2) if a_fmt_bp_ex else None

    w_fmt_bp = mx_specs.get("w_elem_format_bp")
    a_fmt_bp_os = mx_specs.get("a_elem_format_bp_os")
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt_bp}, "w_elem_format",
                               block_size, "round_mx_weight_grad_input", block_axis=0) if w_fmt_bp else None
    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_os}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_input", block_axis=-1) if a_fmt_bp_os else None

    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")

    return OpQuantConfig(
        storage=storage, input=input_mx, weight=weight_mx,
        grad_output=go_elem, grad_input=gi_elem,
        grad_weight=gw_elem, grad_bias=gb_elem,
        input_gw=input_gw_mx, grad_output_gw=grad_output_gw_mx,
        weight_gi=weight_gi_mx, grad_output_gi=grad_output_gi_mx,
    )


def _matmul_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem):
    """Build OpQuantConfig with backward fields for matmul."""
    a_fmt_bp = mx_specs.get("a_elem_format_bp")
    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp}, "a_elem_format",
                              block_size, "round_mx_input_grad_input", block_axis=-2) if a_fmt_bp else None

    a_fmt_bp_os = mx_specs.get("a_elem_format_bp_os")
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_os}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_input", block_axis=-2) if a_fmt_bp_os else None

    w_fmt_bp = mx_specs.get("w_elem_format_bp")
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt_bp}, "w_elem_format",
                               block_size, "round_mx_input_grad_input", block_axis=-1) if w_fmt_bp else None

    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_os}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_input", block_axis=-1) if a_fmt_bp_os else None

    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")

    return OpQuantConfig(
        storage=storage, input=input_mx, weight=weight_mx,
        grad_output=go_elem, grad_input=gi_elem,
        grad_weight=gw_elem, grad_bias=gb_elem,
        input_gw=input_gw_mx, grad_output_gw=grad_output_gw_mx,
        weight_gi=weight_gi_mx, grad_output_gi=grad_output_gi_mx,
    )


def _conv_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem):
    """Build OpQuantConfig with backward fields for conv."""
    a_fmt = mx_specs.get("a_elem_format")
    w_fmt = mx_specs.get("w_elem_format")

    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                              block_size, "round_mx_output", block_axis=0) if a_fmt else None
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=0) if a_fmt else None
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt}, "w_elem_format",
                               block_size, "round_mx_output", block_axis=0) if w_fmt else None
    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=1) if a_fmt else None

    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")

    return OpQuantConfig(
        storage=storage, input=input_mx, weight=weight_mx,
        grad_output=go_elem, grad_input=gi_elem,
        grad_weight=gw_elem, grad_bias=gb_elem,
        input_gw=input_gw_mx, grad_output_gw=grad_output_gw_mx,
        weight_gi=weight_gi_mx, grad_output_gi=grad_output_gi_mx,
    )


def _conv_transpose_backward_cfg(mx_specs, block_size, storage, input_mx, weight_mx, go_elem):
    """Build OpQuantConfig with backward fields for conv_transpose."""
    a_fmt = mx_specs.get("a_elem_format")
    w_fmt = mx_specs.get("w_elem_format")

    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                              block_size, "round_mx_output", block_axis=0) if a_fmt else None
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=0) if a_fmt else None
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt}, "w_elem_format",
                               block_size, "round_mx_output", block_axis=1) if w_fmt else None
    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=1) if a_fmt else None

    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")

    return OpQuantConfig(
        storage=storage, input=input_mx, weight=weight_mx,
        grad_output=go_elem, grad_input=gi_elem,
        grad_weight=gw_elem, grad_bias=gb_elem,
        input_gw=input_gw_mx, grad_output_gw=grad_output_gw_mx,
        weight_gi=weight_gi_mx, grad_output_gi=grad_output_gi_mx,
    )


# ---------------------------------------------------------------------------
# Norm operator config adapter
# ---------------------------------------------------------------------------

def norm_config_from_mx_specs(mx_specs: dict, op_type: str = "batch_norm"):
    """Convert an mx_specs dict to (OpQuantConfig, inner_scheme, quantize_backprop) for norm ops.

    Args:
        mx_specs: Dict from golden .pt file or mx_specs default dict.
        op_type: "batch_norm", "layer_norm", "group_norm", or "rms_norm".

    Returns:
        Tuple of (OpQuantConfig, inner_scheme or None, quantize_backprop bool).
    """
    quantize_backprop = mx_specs.get("quantize_backprop", True)

    inner_scheme = _elem_scheme(mx_specs, "round_output")

    if not quantize_backprop:
        return OpQuantConfig(input=inner_scheme, weight=inner_scheme, bias=inner_scheme), \
               inner_scheme, quantize_backprop

    return OpQuantConfig(
        input=inner_scheme, weight=inner_scheme, bias=inner_scheme,
        grad_output=inner_scheme,
    ), inner_scheme, quantize_backprop


# ---------------------------------------------------------------------------
# Activation / Softmax / Pool config adapters
# ---------------------------------------------------------------------------

def activation_config_from_mx_specs(mx_specs: dict):
    """Convert an mx_specs dict to OpQuantConfig for activation ops.

    Returns:
        OpQuantConfig with input/grad_input fields.
    """
    if mx_specs is None:
        return OpQuantConfig()
    quantize_backprop = mx_specs.get("quantize_backprop", True)
    inner_scheme = _elem_scheme(mx_specs, "round_output")
    bw = inner_scheme if (inner_scheme is not None and quantize_backprop) else None
    return OpQuantConfig(input=inner_scheme, grad_input=bw)


def softmax_config_from_mx_specs(mx_specs: dict):
    """Convert an mx_specs dict to (OpQuantConfig, softmax_exp2) for softmax.

    Returns:
        Tuple of (OpQuantConfig, softmax_exp2).
    """
    if mx_specs is None:
        return OpQuantConfig(), False
    quantize_backprop = mx_specs.get("quantize_backprop", True)
    softmax_exp2 = mx_specs.get("softmax_exp2", False)
    inner_scheme = _elem_scheme(mx_specs, "round_output")
    bw = inner_scheme if (inner_scheme is not None and quantize_backprop) else None
    return OpQuantConfig(input=inner_scheme, grad_input=bw), softmax_exp2


def pool_config_from_mx_specs(mx_specs: dict):
    """Convert an mx_specs dict to OpQuantConfig for pool ops.

    Returns:
        OpQuantConfig with input/grad_input fields.
    """
    return activation_config_from_mx_specs(mx_specs)


# ---------------------------------------------------------------------------
# SIMD / Elemwise config adapter
# ---------------------------------------------------------------------------

def simd_config_from_mx_specs(mx_specs: dict):
    """Convert an mx_specs dict to (inner_scheme, quantize_backprop) for SIMD ops.

    Returns:
        Tuple of (inner_scheme or None, quantize_backprop bool).
    """
    if mx_specs is None:
        return None, True
    quantize_backprop = mx_specs.get("quantize_backprop", True)
    inner_scheme = _elem_scheme(mx_specs, "round_output")
    return inner_scheme, quantize_backprop
