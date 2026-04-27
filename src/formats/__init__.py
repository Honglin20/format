"""
Format package: quantization format abstractions.

- FormatBase: abstract base class with Strategy-pattern quantize/export_onnx
- FPFormat: MX sub-byte floating-point formats (fp8_e4m3, fp6_e3m2, ...)
- IntFormat: sign-magnitude integer formats (int8, int4, int2)
- BFloat16Format, Float16Format: standard float formats
- NF4Format: QLoRA-style lookup-table format
- Registry: global name→format mapping + convenience factories
"""

from .base import FormatBase
from .fp_formats import FPFormat
from .int_formats import IntFormat
from .bf16_fp16 import BFloat16Format, Float16Format
from .lookup_formats import NF4Format
from .registry import (
    FORMAT_REGISTRY,
    register_format,
    register_float_format,
    register_int_format,
    get_format,
)

__all__ = [
    "FormatBase",
    "FPFormat",
    "IntFormat",
    "BFloat16Format",
    "Float16Format",
    "NF4Format",
    "FORMAT_REGISTRY",
    "register_format",
    "register_float_format",
    "register_int_format",
    "get_format",
]
