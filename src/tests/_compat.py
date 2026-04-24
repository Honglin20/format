"""
Internal compat helpers for tests only.

Provides scheme_from_mx_specs() to convert golden-reference mx_specs dicts
into QuantScheme objects. This is the ONLY place in src/ that should
reference mx_specs-style dicts — all other test code uses QuantScheme.
"""
from dataclasses import dataclass

from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
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
