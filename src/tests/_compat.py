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
            Use True for quantize_mx golden tests, False for elemwise/bfloat.
            This matches the old code: quantize_elemwise_op uses bfloat format,
            while quantize_mx_op uses the elem_format override.
    """
    if mx_specs is None:
        return None

    fmt = _format_from_mx_specs_dict(mx_specs, use_mx_format=use_mx_format)
    if fmt is None:
        return None

    # Determine granularity
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
    """Derive a FormatBase from an mx_specs dict.

    Args:
        mx_specs: Dict from golden .pt file.
        use_mx_format: If True, check w_elem_format/a_elem_format first.
    """
    if use_mx_format:
        # quantize_mx_op uses elem_format, which maps to w_elem_format/a_elem_format
        for key in ("w_elem_format", "a_elem_format"):
            fmt_name = mx_specs.get(key)
            if fmt_name and isinstance(fmt_name, str):
                return FormatBase.from_str(fmt_name)

    # elemwise/bfloat quantization uses bfloat/fp
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

    # No bfloat/fp — fall back to elem_format keys
    if not use_mx_format:
        for key in ("w_elem_format", "a_elem_format"):
            fmt_name = mx_specs.get(key)
            if fmt_name and isinstance(fmt_name, str):
                return FormatBase.from_str(fmt_name)

    return None


# ---------------------------------------------------------------------------
# OpQuantConfig adapter (operator-level)
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
    bs = block_size if block_size > 0 else 32  # mx default block_size=32
    round_mode = mx_specs.get(round_key, mx_specs.get("round", "nearest"))
    return QuantScheme(format=fmt, granularity=GranularitySpec.per_block(bs, axis=block_axis),
                       round_mode=round_mode)


def op_config_from_mx_specs(mx_specs: dict, op_type: str = "linear") -> OpQuantConfig:
    """Convert an mx_specs dict to an OpQuantConfig for equivalence tests.

    Only for test use — maps mx_specs keys to OpQuantConfig fields following
    the conventions of mx/linear.py (op_type="linear"), mx/matmul.py
    (op_type="matmul"), mx/convolution.py (op_type="conv"), or
    mx/transpose_convolution.py (op_type="conv_transpose").

    Args:
        mx_specs: Dict from golden .pt file or mx_specs default dict.
        op_type: "linear", "matmul", "conv", or "conv_transpose" — affects
            forward MX axes and backward gemm conventions.

    Returns:
        OpQuantConfig with populated scheme pipelines.
    """
    block_size = mx_specs.get("block_size", 0)
    quantize_backprop = mx_specs.get("quantize_backprop", True)

    # --- Forward pipelines ---
    # input/in1: elemwise cast → MX block quant
    input_elem = _elem_scheme(mx_specs, "round_output")
    # For linear/matmul: in1 MX along axis=-1
    # For conv/conv_transpose: in1 MX along axis=1 (channel dim)
    input_mx_axis = 1 if op_type in ("conv", "conv_transpose") else -1
    input_mx = _mx_scheme(mx_specs, "a_elem_format", block_size, "round_mx_output",
                           block_axis=input_mx_axis)
    input_pipeline = tuple(s for s in [input_elem, input_mx] if s is not None)

    # weight/in2: elemwise cast → MX block quant
    weight_elem = _elem_scheme(mx_specs, "round_weight")
    # For linear: weight MX along axis=-1
    # For matmul: in2 MX along axis=-2
    # For conv: weight MX along axis=1 (out_channels dim, same as input)
    # For conv_transpose: weight MX along axis=0 (in_channels dim)
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
    weight_pipeline = tuple(s for s in [weight_elem, weight_mx] if s is not None)

    # bias: elemwise cast only (no MX)
    bias_elem = _elem_scheme(mx_specs, "round_weight")
    bias_pipeline = (bias_elem,) if bias_elem is not None else ()

    # output: linear has 2 elemwise casts (post-matmul + post-bias),
    # matmul has 1 or 2 (post-matmul, then post-bias if bias present)
    output_elem = _elem_scheme(mx_specs, "round_output")
    if output_elem is not None:
        if op_type == "linear":
            output_pipeline = (output_elem, output_elem)
        else:
            output_pipeline = (output_elem, output_elem)
    else:
        output_pipeline = ()

    if not quantize_backprop:
        return OpQuantConfig(
            input=input_pipeline, weight=weight_pipeline,
            bias=bias_pipeline, output=output_pipeline,
        )

    # --- Backward pipelines ---
    # grad_output: elemwise cast
    go_elem = _elem_scheme(mx_specs, "round_grad_input")
    go_pipeline = (go_elem,) if go_elem is not None else ()

    if op_type == "linear":
        return _linear_backward_pipelines(
            mx_specs, block_size, input_elem, go_elem, output_pipeline,
            input_pipeline, weight_pipeline, bias_pipeline, go_pipeline,
        )
    elif op_type == "matmul":
        return _matmul_backward_pipelines(
            mx_specs, block_size, input_elem, go_elem, output_pipeline,
            input_pipeline, weight_pipeline, bias_pipeline, go_pipeline,
        )
    elif op_type == "conv":
        return _conv_backward_pipelines(
            mx_specs, block_size, input_elem, go_elem, output_pipeline,
            input_pipeline, weight_pipeline, bias_pipeline, go_pipeline,
        )
    elif op_type == "conv_transpose":
        return _conv_transpose_backward_pipelines(
            mx_specs, block_size, input_elem, go_elem, output_pipeline,
            input_pipeline, weight_pipeline, bias_pipeline, go_pipeline,
        )
    else:
        raise ValueError(f"Unknown op_type: {op_type}")


def _linear_backward_pipelines(mx_specs, block_size, input_elem, go_elem,
                                output_pipeline, input_pipeline, weight_pipeline,
                                bias_pipeline, go_pipeline):
    """Build backward OpQuantConfig fields for linear operator."""
    # grad_weight gemm: input_gw (axis=-2) + grad_output_gw (axis=-2)
    a_fmt_bp = mx_specs.get("a_elem_format_bp")
    a_fmt_bp_ex = mx_specs.get("a_elem_format_bp_ex") or a_fmt_bp
    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_ex}, "a_elem_format",
                              block_size, "round_mx_input_grad_weight", block_axis=-2) if a_fmt_bp_ex else None
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_ex}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_weight", block_axis=-2) if a_fmt_bp_ex else None
    input_gw_pipeline = (input_gw_mx,) if input_gw_mx is not None else ()
    grad_output_gw_pipeline = (grad_output_gw_mx,) if grad_output_gw_mx is not None else ()

    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gw_pipeline = (gw_elem,) if gw_elem is not None else ()

    # grad_input gemm: weight_gi (axis=0) + grad_output_gi (axis=-1)
    w_fmt_bp = mx_specs.get("w_elem_format_bp")
    a_fmt_bp_os = mx_specs.get("a_elem_format_bp_os")
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt_bp}, "w_elem_format",
                               block_size, "round_mx_weight_grad_input", block_axis=0) if w_fmt_bp else None
    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_os}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_input", block_axis=-1) if a_fmt_bp_os else None
    weight_gi_pipeline = (weight_gi_mx,) if weight_gi_mx is not None else ()
    grad_output_gi_pipeline = (grad_output_gi_mx,) if grad_output_gi_mx is not None else ()

    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gi_pipeline = (gi_elem,) if gi_elem is not None else ()

    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_pipeline = (gb_elem,) if gb_elem is not None else ()

    return OpQuantConfig(
        input=input_pipeline, weight=weight_pipeline,
        bias=bias_pipeline, output=output_pipeline,
        grad_output=go_pipeline, grad_input=gi_pipeline,
        grad_weight=gw_pipeline, grad_bias=gb_pipeline,
        input_gw=input_gw_pipeline, grad_output_gw=grad_output_gw_pipeline,
        weight_gi=weight_gi_pipeline, grad_output_gi=grad_output_gi_pipeline,
    )


def _matmul_backward_pipelines(mx_specs, block_size, input_elem, go_elem,
                                output_pipeline, input_pipeline, weight_pipeline,
                                bias_pipeline, go_pipeline):
    """Build backward OpQuantConfig fields for matmul operator.

    Matmul backward maps to different axes than linear:
    - grad_in1 = grad_out @ in2^T: in2 needs MX along axis=-1, grad_out along axis=-1
    - grad_in2 = in1^T @ grad_out: in1 needs MX along axis=-2, grad_out along axis=-2
    """
    # in1 for grad_in2 gemm: MX along axis=-2
    a_fmt_bp = mx_specs.get("a_elem_format_bp")
    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp}, "a_elem_format",
                              block_size, "round_mx_input_grad_input", block_axis=-2) if a_fmt_bp else None
    input_gw_pipeline = (input_gw_mx,) if input_gw_mx is not None else ()

    # grad_out for grad_in2 gemm: MX along axis=-2
    a_fmt_bp_os = mx_specs.get("a_elem_format_bp_os")
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_os}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_input", block_axis=-2) if a_fmt_bp_os else None
    grad_output_gw_pipeline = (grad_output_gw_mx,) if grad_output_gw_mx is not None else ()

    # in2 for grad_in1 gemm: MX along axis=-1
    w_fmt_bp = mx_specs.get("w_elem_format_bp")
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt_bp}, "w_elem_format",
                               block_size, "round_mx_input_grad_input", block_axis=-1) if w_fmt_bp else None
    weight_gi_pipeline = (weight_gi_mx,) if weight_gi_mx is not None else ()

    # grad_out for grad_in1 gemm: MX along axis=-1
    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt_bp_os}, "a_elem_format",
                                     block_size, "round_mx_grad_output_grad_input", block_axis=-1) if a_fmt_bp_os else None
    grad_output_gi_pipeline = (grad_output_gi_mx,) if grad_output_gi_mx is not None else ()

    # grad_in1 / grad_in2 exit: elemwise cast
    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gi_pipeline = (gi_elem,) if gi_elem is not None else ()

    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gw_pipeline = (gw_elem,) if gw_elem is not None else ()

    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_pipeline = (gb_elem,) if gb_elem is not None else ()

    return OpQuantConfig(
        input=input_pipeline, weight=weight_pipeline,
        bias=bias_pipeline, output=output_pipeline,
        grad_output=go_pipeline, grad_input=gi_pipeline,
        grad_weight=gw_pipeline, grad_bias=gb_pipeline,
        input_gw=input_gw_pipeline, grad_output_gw=grad_output_gw_pipeline,
        weight_gi=weight_gi_pipeline, grad_output_gi=grad_output_gi_pipeline,
    )


def _conv_backward_pipelines(mx_specs, block_size, input_elem, go_elem,
                              output_pipeline, input_pipeline, weight_pipeline,
                              bias_pipeline, go_pipeline):
    """Build backward OpQuantConfig fields for conv operator.

    Conv backward uses a_elem_format/w_elem_format (NOT _bp variants).
    Axes differ from linear/matmul:
    - grad_weight: input MX along axis=0 (batch), grad_output MX along axis=0
    - grad_input: weight MX along axis=0 (out_channels), grad_output MX along axis=1
    """
    a_fmt = mx_specs.get("a_elem_format")
    w_fmt = mx_specs.get("w_elem_format")

    # input for grad_weight: MX along axis=0 (batch dim)
    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                              block_size, "round_mx_output", block_axis=0) if a_fmt else None
    input_gw_pipeline = (input_gw_mx,) if input_gw_mx is not None else ()

    # grad_output for grad_weight: MX along axis=0 (batch dim)
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=0) if a_fmt else None
    grad_output_gw_pipeline = (grad_output_gw_mx,) if grad_output_gw_mx is not None else ()

    # weight for grad_input: MX along axis=0 (out_channels)
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt}, "w_elem_format",
                               block_size, "round_mx_output", block_axis=0) if w_fmt else None
    weight_gi_pipeline = (weight_gi_mx,) if weight_gi_mx is not None else ()

    # grad_output for grad_input: MX along axis=1 (out_channels)
    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=1) if a_fmt else None
    grad_output_gi_pipeline = (grad_output_gi_mx,) if grad_output_gi_mx is not None else ()

    # Exit elemwise
    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gi_pipeline = (gi_elem,) if gi_elem is not None else ()

    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gw_pipeline = (gw_elem,) if gw_elem is not None else ()

    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_pipeline = (gb_elem,) if gb_elem is not None else ()

    return OpQuantConfig(
        input=input_pipeline, weight=weight_pipeline,
        bias=bias_pipeline, output=output_pipeline,
        grad_output=go_pipeline, grad_input=gi_pipeline,
        grad_weight=gw_pipeline, grad_bias=gb_pipeline,
        input_gw=input_gw_pipeline, grad_output_gw=grad_output_gw_pipeline,
        weight_gi=weight_gi_pipeline, grad_output_gi=grad_output_gi_pipeline,
    )


def _conv_transpose_backward_pipelines(mx_specs, block_size, input_elem, go_elem,
                                        output_pipeline, input_pipeline, weight_pipeline,
                                        bias_pipeline, go_pipeline):
    """Build backward OpQuantConfig fields for conv_transpose operator.

    ConvTranspose backward differs from conv:
    - grad_weight: same as conv (input MX axis=0, grad_output MX axis=0)
      but conv_weight() receives (grad_output, weight.shape, input) —
      note the swapped order vs conv's (input, weight.shape, grad_output)
    - grad_input: uses F.conv2d (not conv_transpose), weight MX along axis=1
      (out_channels/groups dim in transposed weight layout)
    """
    a_fmt = mx_specs.get("a_elem_format")
    w_fmt = mx_specs.get("w_elem_format")

    # input for grad_weight: MX along axis=0 (batch dim) — same as conv
    input_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                              block_size, "round_mx_output", block_axis=0) if a_fmt else None
    input_gw_pipeline = (input_gw_mx,) if input_gw_mx is not None else ()

    # grad_output for grad_weight: MX along axis=0 (batch dim) — same as conv
    grad_output_gw_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=0) if a_fmt else None
    grad_output_gw_pipeline = (grad_output_gw_mx,) if grad_output_gw_mx is not None else ()

    # weight for grad_input: MX along axis=1 (out_channels/groups dim)
    # Differs from conv which uses axis=0
    weight_gi_mx = _mx_scheme({**mx_specs, "w_elem_format": w_fmt}, "w_elem_format",
                               block_size, "round_mx_output", block_axis=1) if w_fmt else None
    weight_gi_pipeline = (weight_gi_mx,) if weight_gi_mx is not None else ()

    # grad_output for grad_input: MX along axis=1 (out_channels dim) — same as conv
    grad_output_gi_mx = _mx_scheme({**mx_specs, "a_elem_format": a_fmt}, "a_elem_format",
                                     block_size, "round_mx_output", block_axis=1) if a_fmt else None
    grad_output_gi_pipeline = (grad_output_gi_mx,) if grad_output_gi_mx is not None else ()

    # Exit elemwise
    gi_elem = _elem_scheme(mx_specs, "round_grad_input")
    gi_pipeline = (gi_elem,) if gi_elem is not None else ()

    gw_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gw_pipeline = (gw_elem,) if gw_elem is not None else ()

    gb_elem = _elem_scheme(mx_specs, "round_grad_weight")
    gb_pipeline = (gb_elem,) if gb_elem is not None else ()

    return OpQuantConfig(
        input=input_pipeline, weight=weight_pipeline,
        bias=bias_pipeline, output=output_pipeline,
        grad_output=go_pipeline, grad_input=gi_pipeline,
        grad_weight=gw_pipeline, grad_bias=gb_pipeline,
        input_gw=input_gw_pipeline, grad_output_gw=grad_output_gw_pipeline,
        weight_gi=weight_gi_pipeline, grad_output_gi=grad_output_gi_pipeline,
    )


# ---------------------------------------------------------------------------
# Norm operator config adapter
# ---------------------------------------------------------------------------

def norm_config_from_mx_specs(mx_specs: dict, op_type: str = "batch_norm"):
    """Convert an mx_specs dict to (OpQuantConfig, inner_scheme, quantize_backprop) for norm ops.

    Norm operators (BatchNorm, LayerNorm, GroupNorm, RMSNorm) use only elemwise
    quantization — no MX block quantization. The OpQuantConfig provides
    entry quantization only (input/weight/bias/grad_output), while inner_scheme
    provides the QuantScheme used for all intermediate vec_op computations.
    Exit quantization (output/grad_input/grad_weight/grad_bias) is empty
    because mx norm operators have no post-vec_op output quantization.

    Args:
        mx_specs: Dict from golden .pt file or mx_specs default dict.
        op_type: "batch_norm", "layer_norm", "group_norm", or "rms_norm".

    Returns:
        Tuple of (OpQuantConfig, inner_scheme QuantScheme or None, quantize_backprop bool).
    """
    quantize_backprop = mx_specs.get("quantize_backprop", True)

    # Inner scheme: same elemwise format used for all intermediate vec_ops
    # and for entry quantization of input/weight/bias/grad_output.
    # All vec_ops use mx_specs['round'] (default round mode).
    inner_scheme = _elem_scheme(mx_specs, "round_output")

    # Forward entry schemes — same format as inner_scheme, same round mode
    # mx norm forward: vec_quantize(input), vec_quantize(weight), vec_quantize(bias)
    # No output quantization after _norm_forward (already quantized by vec_add)
    input_pipeline = (inner_scheme,) if inner_scheme is not None else ()
    weight_pipeline = (inner_scheme,) if inner_scheme is not None else ()
    bias_pipeline = (inner_scheme,) if inner_scheme is not None else ()
    output_pipeline = ()

    if not quantize_backprop:
        return OpQuantConfig(
            input=input_pipeline, weight=weight_pipeline,
            bias=bias_pipeline, output=output_pipeline,
        ), inner_scheme, quantize_backprop

    # Backward entry for grad_output: vec_quantize(grad_output, mx_specs)
    # Same format/round as inner_scheme since mx uses mx_specs['round']
    go_pipeline = (inner_scheme,) if inner_scheme is not None else ()

    # No exit quantization — mx norm backward has no post-vec_op quantization
    return OpQuantConfig(
        input=input_pipeline, weight=weight_pipeline,
        bias=bias_pipeline, output=output_pipeline,
        grad_output=go_pipeline,
    ), inner_scheme, quantize_backprop


# ---------------------------------------------------------------------------
# Activation / Softmax / Pool config adapters
# ---------------------------------------------------------------------------

def activation_config_from_mx_specs(mx_specs: dict):
    """Convert an mx_specs dict to OpQuantConfig for activation ops.

    Activation operators use only elemwise quantization via inner_scheme.
    The returned OpQuantConfig has input=(inner_scheme,) for forward and
    grad_input=(inner_scheme,) for backward QAT (empty if quantize_backprop=False).

    Returns:
        OpQuantConfig with populated input/grad_input fields.
    """
    if mx_specs is None:
        return OpQuantConfig()
    quantize_backprop = mx_specs.get("quantize_backprop", True)
    inner_scheme = _elem_scheme(mx_specs, "round_output")
    fwd_pipeline = (inner_scheme,) if inner_scheme is not None else ()
    bw_pipeline = (inner_scheme,) if (inner_scheme is not None and quantize_backprop) else ()
    return OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline)


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
    fwd_pipeline = (inner_scheme,) if inner_scheme is not None else ()
    bw_pipeline = (inner_scheme,) if (inner_scheme is not None and quantize_backprop) else ()
    return OpQuantConfig(input=fwd_pipeline, grad_input=bw_pipeline), softmax_exp2


def pool_config_from_mx_specs(mx_specs: dict):
    """Convert an mx_specs dict to OpQuantConfig for pool ops.

    Returns:
        OpQuantConfig with populated input/grad_input fields.
    """
    return activation_config_from_mx_specs(mx_specs)


# ---------------------------------------------------------------------------
# SIMD / Elemwise config adapter
# ---------------------------------------------------------------------------


def simd_config_from_mx_specs(mx_specs: dict):
    """Convert an mx_specs dict to (inner_scheme, quantize_backprop) for SIMD ops.

    SIMD operators use elemwise quantization via inner_scheme.
    Same pattern as activation ops.

    Returns:
        Tuple of (inner_scheme QuantScheme or None, quantize_backprop bool).
    """
    if mx_specs is None:
        return None, True
    quantize_backprop = mx_specs.get("quantize_backprop", True)
    inner_scheme = _elem_scheme(mx_specs, "round_output")
    return inner_scheme, quantize_backprop
