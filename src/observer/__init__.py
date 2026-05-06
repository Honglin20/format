"""Cross-cutting observer infrastructure for quantized operator event collection.

ObservableMixin is the base for all QuantizedXxx modules in ops/.
ObserverBase + SliceAwareObserver are base classes for analysis observers.
QuantEvent is the event dataclass emitted at each quantization point.

This package is independent of both ops/ (which consumes ObservableMixin)
and analysis/ (which consumes ObserverBase + QuantEvent), so neither
layer has a reverse dependency on the other.
"""

from .events import QuantEvent
from .mixin import ObservableMixin
from .observer import ObserverBase, SliceAwareObserver, SliceKey

__all__ = [
    "QuantEvent",
    "ObservableMixin",
    "ObserverBase",
    "SliceAwareObserver",
    "SliceKey",
]
