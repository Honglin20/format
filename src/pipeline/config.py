from typing import Any, Dict

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform, TransformBase
from src.transform.hadamard import HadamardTransform


def _resolve_granularity(desc: Dict[str, Any]) -> GranularitySpec:
    """Convert a search-space granularity descriptor to GranularitySpec."""
    mode = desc.get("granularity")
    if mode is None:
        raise ValueError("descriptor must contain 'granularity' key")
    if not isinstance(mode, str):
        raise TypeError(
            f"'granularity' must be a string, got {type(mode).__name__}"
        )
    if mode == "per_tensor":
        return GranularitySpec.per_tensor()
    elif mode == "per_channel":
        axis = desc.get("axis", -1)
        if not isinstance(axis, int):
            raise TypeError(f"'axis' must be an int, got {type(axis).__name__}")
        return GranularitySpec.per_channel(axis=axis)
    elif mode == "per_block":
        block_size = desc.get("block_size")
        if block_size is None:
            raise ValueError("per_block granularity requires 'block_size' in descriptor")
        if not isinstance(block_size, int):
            raise TypeError(
                f"'block_size' must be an int, got {type(block_size).__name__}"
            )
        axis = desc.get("axis", -1)
        if not isinstance(axis, int):
            raise TypeError(f"'axis' must be an int, got {type(axis).__name__}")
        return GranularitySpec.per_block(size=block_size, axis=axis)
    else:
        raise ValueError(f"Unknown granularity: {mode}")


def _resolve_transform(desc: Dict[str, Any]) -> TransformBase:
    """Convert a search-space transform descriptor to TransformBase."""
    tx = desc.get("transform")
    if tx is None:
        return IdentityTransform()
    if isinstance(tx, TransformBase):
        return tx
    if isinstance(tx, str):
        if tx == "hadamard":
            return HadamardTransform()
        raise ValueError(f"Unknown transform: {tx}")
    raise TypeError(
        f"'transform' must be a string, TransformBase, or None, "
        f"got {type(tx).__name__}"
    )


def resolve_config(desc: Dict[str, Any]) -> OpQuantConfig:
    """Convert a search-space descriptor dict to OpQuantConfig.

    Args:
        desc: Dict with keys:
            - format (str): Format name e.g. "int8", "fp8_e4m3", "nf4"
            - granularity (str): "per_tensor" | "per_channel" | "per_block"
            - axis (int): Channel/block axis (default -1)
            - block_size (int): Required for per_block
            - transform (str | TransformBase | None): "hadamard" or instance
            - weight_only (bool): If True, only weight is quantized

    Returns:
        OpQuantConfig with input, weight, output set
        (or just weight if weight_only).
    """
    fmt_name = desc.get("format")
    if fmt_name is None:
        raise ValueError("descriptor must contain 'format' key")
    if not isinstance(fmt_name, str):
        raise TypeError(
            f"'format' must be a string, got {type(fmt_name).__name__}"
        )
    fmt = FormatBase.from_str(fmt_name)
    granularity = _resolve_granularity(desc)
    transform = _resolve_transform(desc)
    scheme = QuantScheme(format=fmt, granularity=granularity, transform=transform)

    weight_only = desc.get("weight_only", False)
    if not isinstance(weight_only, bool):
        raise TypeError(
            f"'weight_only' must be a bool, got {type(weight_only).__name__}"
        )
    if weight_only:
        return OpQuantConfig(weight=scheme)
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)
