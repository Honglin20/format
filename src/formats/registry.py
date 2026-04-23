"""
Format registry: global name→FormatBase mapping.

Replaces ElemFormat.from_str(). Supports aliases and custom format registration.
"""
from typing import Dict, Optional
from .base import FormatBase

# Global registry: canonical name → FormatBase instance
FORMAT_REGISTRY: Dict[str, FormatBase] = {}

# Alias map: alias → canonical name
_ALIASES: Dict[str, str] = {}


def register_format(name: str, fmt: FormatBase, aliases: list = None,
                    overwrite: bool = False):
    """Register a format in the global registry.

    Args:
        name: Canonical format name (e.g., "fp8_e4m3")
        fmt: FormatBase instance
        aliases: Optional list of alternative names (e.g., ["fp4"] for "fp4_e2m1")
        overwrite: If True, silently replace existing entry. If False (default),
                   raise ValueError if name already exists.
    """
    if not overwrite and name in FORMAT_REGISTRY:
        raise ValueError(
            f"Format {name!r} already registered. Use overwrite=True to replace."
        )
    FORMAT_REGISTRY[name] = fmt
    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name


def get_format(name: str) -> FormatBase:
    """Look up a format by name. Case-insensitive. Supports canonical names and aliases."""
    name = name.lower()
    if name in FORMAT_REGISTRY:
        return FORMAT_REGISTRY[name]
    if name in _ALIASES:
        return FORMAT_REGISTRY[_ALIASES[name]]
    raise ValueError(f"Unknown format: {name!r}")


def _init_default_formats():
    """Register all default formats. Called once at import time."""
    from .int_formats import IntFormat
    from .fp_formats import FPFormat
    from .bf16_fp16 import BFloat16Format, Float16Format

    # Integer formats
    for bits, name in [(8, "int8"), (4, "int4"), (2, "int2")]:
        register_format(name, IntFormat(bits=bits, name=name))

    # Float formats (MX sub-byte)
    float_formats = [
        ("fp8_e5m2", 5, 4, None),
        ("fp8_e4m3", 4, 5, 448.0),  # Custom max_norm: 2**8 * 1.75
        ("fp6_e3m2", 3, 4, None),
        ("fp6_e2m3", 2, 5, None),
        ("fp4_e2m1", 2, 3, None),
    ]
    for name, ebits, mbits, max_norm_override in float_formats:
        register_format(name, FPFormat(name=name, ebits=ebits, mbits=mbits,
                                       max_norm_override=max_norm_override))

    # Aliases — overwrite=True because the canonical name already exists
    register_format("fp4_e2m1", FORMAT_REGISTRY["fp4_e2m1"], aliases=["fp4"],
                    overwrite=True)

    # Standard formats
    register_format("float16", Float16Format(), aliases=["fp16"])
    register_format("bfloat16", BFloat16Format(), aliases=["bf16"])


# Auto-initialize on import
_init_default_formats()
