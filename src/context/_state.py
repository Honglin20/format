import contextvars
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.scheme.op_config import OpQuantConfig

# Singleton used as sentinel for "no quantization" equality checks in _patches.py.
_EMPTY_CFG = OpQuantConfig()


@dataclass
class _CtxState:
    cfg: OpQuantConfig
    op_cfgs: Dict[str, OpQuantConfig] = field(default_factory=dict)
    observers: List = field(default_factory=list)

    def resolve(self, op_name: str) -> OpQuantConfig:
        """Return per-op cfg if overridden, else default cfg."""
        if self.op_cfgs and op_name in self.op_cfgs:
            return self.op_cfgs[op_name]
        return self.cfg


_ctx_state: contextvars.ContextVar[Optional[_CtxState]] = contextvars.ContextVar(
    "quant_ctx_state", default=None
)
