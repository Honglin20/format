"""
Analysis infrastructure for quantized operators.

Phase 3 lands the skeleton (ObservableMixin + QuantEvent + iter_slices).
Phase 4 adds AnalysisContext and concrete Observer implementations.
"""
from .events import QuantEvent
from .mixin import ObservableMixin
from .observer import ObserverBase, SliceAwareObserver
from .slicing import iter_slices, SliceKey
