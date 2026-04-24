"""
Observer base classes for analysis.

ObserverBase: abstract base for fully custom observers.
SliceAwareObserver: granularity-aware base that auto-slices via iter_slices.
"""
from abc import ABC, abstractmethod
from typing import Tuple

from src.analysis.events import QuantEvent
from src.analysis.slicing import SliceKey, iter_slices


class ObserverBase(ABC):
    """Abstract observer base class. Override on_event for custom logic."""

    @abstractmethod
    def on_event(self, event: QuantEvent) -> None:
        ...

    def report(self) -> dict:
        """Return nested dict: {layer_name: {role: {stage: {slice_key: metric}}}}"""
        return {}

    def reset(self):
        """Clear accumulated state."""


class SliceAwareObserver(ObserverBase):
    """Granularity-aware observer base. Auto-slices via iter_slices.

    Subclasses only need to implement _measure(key, fp32, quant) -> metric_dict.
    """

    def __init__(self):
        self._buffer: dict = {}

    @abstractmethod
    def _measure(self, key: SliceKey, fp32, quant) -> dict:
        """Compute metrics for a single slice. Return e.g. {'qsnr_db': 38.2}."""
        ...

    def on_event(self, event: QuantEvent):
        for key, f, q in iter_slices(
            event.fp32_tensor, event.quant_tensor,
            event.scheme.granularity, event.group_map,
        ):
            metric = self._measure(key, f, q)
            dst = (self._buffer
                   .setdefault(event.layer_name, {})
                   .setdefault(event.role, {})
                   .setdefault(f"{event.stage}[{event.pipeline_index}]", {}))
            dst[key] = metric

    def report(self) -> dict:
        return self._buffer

    def reset(self):
        self._buffer.clear()
