from typing import Any, Dict

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform, TransformBase
from src.transform.hadamard import HadamardTransform


def _resolve_granularity(desc: Dict[str, Any]) -> GranularitySpec:
    """Convert a search-space granularity descriptor to GranularitySpec."""
    mode = desc["granularity"]
    if mode == "per_tensor":
        return GranularitySpec.per_tensor()
    elif mode == "per_channel":
        axis = desc.get("axis", -1)
        return GranularitySpec.per_channel(axis=axis)
    elif mode == "per_block":
        block_size = desc.get("block_size")
        if block_size is None:
            raise ValueError("per_block granularity requires 'block_size' in descriptor")
        axis = desc.get("axis", -1)
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
    if tx == "hadamard":
        return HadamardTransform()
    raise ValueError(f"Unknown transform: {tx}")


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
    fmt = FormatBase.from_str(fmt_name)
    granularity = _resolve_granularity(desc)
    transform = _resolve_transform(desc)
    scheme = QuantScheme(format=fmt, granularity=granularity, transform=transform)

    weight_only = desc.get("weight_only", False)
    if weight_only:
        return OpQuantConfig(weight=scheme)
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)
