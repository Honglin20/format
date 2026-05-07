"""
Shared format parametrization for equivalence and E2E tests.

Provides reusable pytest.param lists so every test file doesn't
duplicate the same MX_SPECS_CONFIGS definitions.

Usage::

    from src.tests._formats import smoke_mx_specs_params, full_mx_specs_params

    class TestSomething:
        @pytest.mark.parametrize("name,mx_specs", smoke_mx_specs_params())
        def test_quick(self, name, mx_specs): ...

        @pytest.mark.parametrize("name,mx_specs", full_mx_specs_params())
        @pytest.mark.slow
        def test_comprehensive(self, name, mx_specs): ...
"""
from __future__ import annotations

from typing import List

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Format constants
# ═══════════════════════════════════════════════════════════════════════════

ALL_MX_ELEM_FORMATS: List[str] = [
    "fp8_e5m2",
    "fp8_e4m3",
    "fp6_e3m2",
    "fp6_e2m3",
    "fp4_e2m1",
    "int8",
    "int4",
    "int2",
]

SMOKE_MX_FORMATS: List[str] = [
    "fp8_e4m3",   # most common MX float
    "int8",        # 8-bit integer
    "int4",        # low-precision (triggers BF16 matmul path in mx)
]

DEFAULT_BLOCK_SIZE = 32

# ═══════════════════════════════════════════════════════════════════════════
# MX specs builders
# ═══════════════════════════════════════════════════════════════════════════


def _format_id(mx_format: str) -> str:
    """Short id fragment for a format string, e.g. 'fp8e4m3' from 'fp8_e4m3'."""
    return mx_format.replace("_", "")


def _build_spec(
    storage: str | None,        # "bf16" | "bf10" | None
    mx_fmt: str | None,         # MX elem format string or None
    quantize_backprop: bool,
    block_size: int,
) -> dict:
    """Build a raw mx_specs dict (to be passed through apply_mx_specs)."""
    specs: dict = {}
    if storage == "bf16":
        specs["bfloat"] = 16
    elif storage == "bf10":
        specs["bfloat"] = 10
    # storage is None → no bfloat key

    if mx_fmt is not None:
        specs["w_elem_format"] = mx_fmt
        specs["a_elem_format"] = mx_fmt
        specs["block_size"] = block_size

    if not quantize_backprop:
        specs["quantize_backprop"] = False

    return specs


def _build_id(storage: str | None, mx_fmt: str | None, qbp: bool) -> str:
    """Build a pytest id string from components.

    Examples: 'bf16', 'bf16+mxfp8e4m3', 'mxfp8e4m3', 'bf16+mxfp8e4m3-ste'
    """
    parts: List[str] = []
    if storage is not None:
        parts.append(storage)
    if mx_fmt is not None:
        parts.append("mx" + _format_id(mx_fmt))
    if not qbp:
        parts.append("ste")
    return "+".join(parts) if parts else "passthrough"


def build_mx_specs_params(
    mx_formats: List[str] | None = None,
    storage_modes: List[str | None] | None = None,
    quantize_backprop_modes: List[bool] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    include_passthrough: bool = True,
) -> List:
    """Build a parametrize-ready list of ``pytest.param(name, specs, id=...)``.

    Args:
        mx_formats: MX elem_format strings to include. None → all 8.
        storage_modes: Storage modes: ``"bf16"``, ``"bf10"``, ``None`` (no storage).
            None → ``["bf16"]``.
        quantize_backprop_modes: QBP flags. None → ``[True]``.
        block_size: MX block size.
        include_passthrough: Append a no-quantization config at the end.
    """
    if mx_formats is None:
        mx_formats = ALL_MX_ELEM_FORMATS
    if storage_modes is None:
        storage_modes = ["bf16"]
    if quantize_backprop_modes is None:
        quantize_backprop_modes = [True]

    params: List = []
    for storage in storage_modes:
        for mx_fmt in mx_formats:
            for qbp in quantize_backprop_modes:
                specs = _build_spec(storage, mx_fmt, qbp, block_size)
                pid = _build_id(storage, mx_fmt, qbp)
                params.append(pytest.param(pid, specs, id=pid))

    if include_passthrough:
        params.append(pytest.param(
            "passthrough", {}, id="passthrough",
        ))

    return params


# ═══════════════════════════════════════════════════════════════════════════
# Convenience presets
# ═══════════════════════════════════════════════════════════════════════════


def smoke_mx_specs_params() -> List:
    """Smoke-test subset: key format × storage combinations, fast enough for CI.

    Covers the critical paths:
    - bfloat16-only (baseline elemwise)
    - bf16 + fp8e4m3 / int8 / int4 (representative MX formats)
    - mxfp8e4m3 with no bf16 storage (pure MX path)
    - passthrough (no quantization)

    Note: STE (quantize_backprop=False) is NOT included because backward
    STE + MX equivalence for Conv ops is not yet validated.
    """
    params: List = []

    # bfloat16 only (no MX block)
    params.append(pytest.param("bf16", {"bfloat": 16}, id="bf16"))

    # bf16 + representative MX formats
    for fmt in SMOKE_MX_FORMATS:
        specs = _build_spec("bf16", fmt, True, DEFAULT_BLOCK_SIZE)
        pid = _build_id("bf16", fmt, True)
        params.append(pytest.param(pid, specs, id=pid))

    # Pure MX (no bfloat storage)
    specs = _build_spec(None, "fp8_e4m3", True, DEFAULT_BLOCK_SIZE)
    pid = _build_id(None, "fp8_e4m3", True)
    params.append(pytest.param(pid, specs, id=pid))

    # Passthrough
    params.append(pytest.param("passthrough", {}, id="passthrough"))

    return params


def full_mx_specs_params() -> List:
    """Full format matrix: all 8 MX formats × bf16, plus key variants.

    Use with ``@pytest.mark.slow`` — this generates ~15 configs.
    """
    return build_mx_specs_params(
        mx_formats=ALL_MX_ELEM_FORMATS,
        storage_modes=["bf16"],
        quantize_backprop_modes=[True],
        block_size=DEFAULT_BLOCK_SIZE,
        include_passthrough=True,
    )


def extended_mx_specs_params() -> List:
    """Extended format matrix including no-storage and STE variants.

    Use with ``@pytest.mark.slow`` — this generates ~23 configs.
    """
    params: List = []

    # bf16 × all 8 MX formats × QBP
    params.extend(build_mx_specs_params(
        mx_formats=ALL_MX_ELEM_FORMATS,
        storage_modes=["bf16"],
        quantize_backprop_modes=[True],
        block_size=DEFAULT_BLOCK_SIZE,
        include_passthrough=False,
    ))

    # No storage × smoke MX formats (pure MX path)
    params.extend(build_mx_specs_params(
        mx_formats=SMOKE_MX_FORMATS,
        storage_modes=[None],
        quantize_backprop_modes=[True],
        block_size=DEFAULT_BLOCK_SIZE,
        include_passthrough=False,
    ))

    # bf16 + STE × smoke MX formats (no backward quantization)
    params.extend(build_mx_specs_params(
        mx_formats=SMOKE_MX_FORMATS,
        storage_modes=["bf16"],
        quantize_backprop_modes=[False],
        block_size=DEFAULT_BLOCK_SIZE,
        include_passthrough=False,
    ))

    # Passthrough
    params.append(pytest.param("passthrough", {}, id="passthrough"))

    return params


# ═══════════════════════════════════════════════════════════════════════════
# Elemwise-only parametrization (norms, activations, softmax, pool, simd)
# ═══════════════════════════════════════════════════════════════════════════


def elemwise_specs_params() -> List:
    """Standard elemwise-only parametrization.

    These operators don't use MX block quantization — they only use
    bfloat/fp elemwise quantization.
    """
    return [
        pytest.param("bf16", {"bfloat": 16}, id="bf16"),
        pytest.param("bf10", {"bfloat": 10}, id="bf10"),
        pytest.param("bf16-ste", {"bfloat": 16, "quantize_backprop": False}, id="bf16-ste"),
        pytest.param("passthrough", {}, id="passthrough"),
    ]
