"""
ObservableMixin: no-op skeleton for quantized operators.

Phase 3: _emit does nothing (early return when _observers is empty).
Phase 4: AnalysisContext will populate _observers to enable event dispatch.
"""
from typing import List, Optional

from torch import Tensor

from src.scheme.quant_scheme import QuantScheme


class ObservableMixin:
    """Mixin for quantized operators to emit analysis events.

    Zero overhead: when no observers are attached, _emit returns immediately.
    Phase 3 lands this as a no-op skeleton; Phase 4 adds AnalysisContext
    to populate _observers and dispatch events.
    """

    _observers: list = []
    _analysis_name: Optional[str] = None

    def _emit(self, role: str, pipeline_index: int, stage: str,
              fp32: Tensor, quant: Tensor, scheme: QuantScheme,
              group_map: Optional[Tensor] = None) -> None:
        """Emit a QuantEvent to all attached observers.

        No-op when no observers are attached (zero overhead).
        """
        if not self._observers:
            return
        from src.analysis.events import QuantEvent
        event = QuantEvent(
            layer_name=self._analysis_name or type(self).__name__,
            role=role,
            pipeline_index=pipeline_index,
            stage=stage,
            fp32_tensor=fp32.detach(),
            quant_tensor=quant.detach(),
            scheme=scheme,
            group_map=group_map.detach() if group_map is not None else None,
        )
        for obs in self._observers:
            obs.on_event(event)
