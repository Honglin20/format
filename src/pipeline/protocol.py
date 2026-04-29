from typing import Any, Dict, Protocol

import torch.nn as nn


class EvalFn(Protocol):
    """User-provided evaluation function.

    Called by ExperimentRunner in three contexts:
    - Calibration: forward side-effects trigger hooks, return value ignored
    - Analysis: forward side-effects trigger observer hooks, return value ignored
    - Evaluation: return value used for fp32 vs quant delta computation
    """

    def __call__(self, model: nn.Module, data: Any) -> Dict[str, float]: ...
