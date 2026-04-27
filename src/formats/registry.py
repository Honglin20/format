"""
Format registry: global name→FormatBase mapping.

Replaces ElemFormat.from_str(). Supports aliases and custom format registration.
Uses lazy initialization to avoid import-time side effects and threading issues.

One-liner factory functions::

    register_float_format("fp5_e3m1", ebits=3, mbits=1)
    register_int_format("int16", bits=16)
    FormatBase.from_str("fp5_e3m1")   # auto-parsed from naming convention
    FormatBase.from_str("int16")       # auto-parsed from naming convention
"""
import re
import threading
from typing import Dict, Optional
from .base import FormatBase

# Global registry: canonical name → FormatBase instance
FORMAT_REGISTRY: Dict[str, FormatBase] = {}

# Alias map: alias → canonical name
_ALIASES: Dict[str, str] = {}

# Guard for lazy initialization and thread safety
_lock = threading.Lock()
_initialized = False

# Regex patterns for auto-parsing format names
_FLOAT_FORMAT_RE = re.compile(r"^fp(\d+)_e(\d+)m(\d+)$")
_INT_FORMAT_RE = re.compile(r"^int(\d+)$")


def _ensure_initialized():
    """Lazily register default formats on first access."""
    global _initialized
    if not _initialized:
        with _lock:
            if not _initialized:
                _init_default_formats()
                _initialized = True


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
    _ensure_initialized()
    if not overwrite and name in FORMAT_REGISTRY:
        raise ValueError(
            f"Format {name!r} already registered. Use overwrite=True to replace."
        )
    FORMAT_REGISTRY[name] = fmt
    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name


def register_float_format(
    name: str,
    ebits: int,
    mbits: int,
    max_norm_override: float = None,
    aliases: list = None,
    overwrite: bool = False,
) -> FormatBase:
    """Register a float format by parameters (no class definition needed).

    Example::

        register_float_format("fp5_e3m1", ebits=3, mbits=1)
        fmt = FormatBase.from_str("fp5_e3m1")

    Args:
        name: Canonical name (e.g. ``"fp5_e3m1"``).
        ebits: Number of exponent bits.
        mbits: Number of mantissa bits (includes implicit leading 1).
        max_norm_override: If None, auto-computed; otherwise explicit max normal.
        aliases: Optional list of alternative names.
        overwrite: If True, silently replace existing entry.

    Returns:
        The registered FPFormat instance.
    """
    from .fp_formats import FPFormat
    if ebits < 0 or mbits < 0:
        raise ValueError(
            f"ebits and mbits must be non-negative, got ebits={ebits}, mbits={mbits}"
        )
    fmt = FPFormat(name=name, ebits=ebits, mbits=mbits, max_norm_override=max_norm_override)
    register_format(name, fmt, aliases=aliases, overwrite=overwrite)
    return fmt


def register_int_format(
    name: str,
    bits: int,
    aliases: list = None,
    overwrite: bool = False,
) -> FormatBase:
    """Register an integer format by bit-width (no class definition needed).

    Example::

        register_int_format("int16", bits=16)
        fmt = FormatBase.from_str("int16")

    Args:
        name: Canonical name (e.g. ``"int16"``).
        bits: Total bit width (must be >= 2).
        aliases: Optional list of alternative names.
        overwrite: If True, silently replace existing entry.

    Returns:
        The registered IntFormat instance.
    """
    from .int_formats import IntFormat
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    fmt = IntFormat(bits=bits, name=name)
    register_format(name, fmt, aliases=aliases, overwrite=overwrite)
    return fmt


def _try_parse_format_name(name: str) -> Optional[FormatBase]:
    """Try to auto-create a format from its naming convention.

    Supported patterns:
        ``fp<total>_e<ebits>_m<actual_mbits>`` → FPFormat (e.g. ``fp5_e3m1``)
            The naming convention uses *actual* mantissa bits (no implicit).
            FPFormat's mbits = sign_bit + implicit_bit + actual_mantissa_bits,
            so we add 2 when constructing (1 sign + 1 implicit).
        ``int<bits>`` → IntFormat (e.g. ``int16``)

    Returns the created FormatBase, or None if the name doesn't match.
    """
    m = _FLOAT_FORMAT_RE.match(name)
    if m:
        total_bits = int(m.group(1))
        ebits = int(m.group(2))
        actual_mbits = int(m.group(3))
        if ebits + actual_mbits + 1 != total_bits:
            raise ValueError(
                f"Cannot parse {name!r}: total bits {total_bits} != "
                f"ebits {ebits} + mbits {actual_mbits} + 1 (sign)"
            )
        # FPFormat mbits = sign(1) + implicit(1) + actual mantissa bits
        code_mbits = actual_mbits + 2
        return register_float_format(name, ebits=ebits, mbits=code_mbits,
                                     overwrite=True)

    m = _INT_FORMAT_RE.match(name)
    if m:
        bits = int(m.group(1))
        return register_int_format(name, bits=bits, overwrite=True)

    return None


def get_format(name: str) -> FormatBase:
    """Look up a format by name. Case-insensitive.

    First checks the registry. If not found, tries to auto-parse the name
    from naming convention (e.g. ``"fp5_e3m1"`` → FPFormat with ebits=3, mbits=1).

    Returns the FormatBase instance.
    """
    _ensure_initialized()
    name = name.lower()
    if name in FORMAT_REGISTRY:
        return FORMAT_REGISTRY[name]
    if name in _ALIASES:
        return FORMAT_REGISTRY[_ALIASES[name]]

    # Auto-parse from naming convention
    fmt = _try_parse_format_name(name)
    if fmt is not None:
        return fmt

    raise ValueError(f"Unknown format: {name!r}")


def _init_default_formats():
    """Register all default formats. Called once on first registry access."""
    from .int_formats import IntFormat
    from .fp_formats import FPFormat
    from .bf16_fp16 import BFloat16Format, Float16Format

    # Integer formats
    for bits, name in [(8, "int8"), (4, "int4"), (2, "int2")]:
        FORMAT_REGISTRY[name] = IntFormat(bits=bits, name=name)

    # Float formats (MX sub-byte)
    float_formats = [
        ("fp8_e5m2", 5, 4, None),
        ("fp8_e4m3", 4, 5, 448.0),  # Custom max_norm: 2**8 * 1.75
        ("fp6_e3m2", 3, 4, None),
        ("fp6_e2m3", 2, 5, None),
        ("fp4_e2m1", 2, 3, None),
    ]
    for name, ebits, mbits, max_norm_override in float_formats:
        FORMAT_REGISTRY[name] = FPFormat(name=name, ebits=ebits, mbits=mbits,
                                         max_norm_override=max_norm_override)

    # Lookup-table formats
    from .lookup_formats import NF4Format
    FORMAT_REGISTRY["nf4"] = NF4Format()

    # Aliases
    _ALIASES["fp4"] = "fp4_e2m1"

    # Standard formats
    FORMAT_REGISTRY["float16"] = Float16Format()
    _ALIASES["fp16"] = "float16"
    FORMAT_REGISTRY["bfloat16"] = BFloat16Format()
    _ALIASES["bf16"] = "bfloat16"
