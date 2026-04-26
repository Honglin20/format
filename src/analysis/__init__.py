"""
Analysis infrastructure for quantized operators.

Phase 4: full AnalysisContext + concrete Observers + Report + Distribution taxonomy.
"""
from .events import QuantEvent
from .mixin import ObservableMixin
from .observer import ObserverBase, SliceAwareObserver
from .slicing import iter_slices, SliceKey
from .observers import DistributionObserver, QSNRObserver, MSEObserver, HistogramObserver
from .context import AnalysisContext
from .report import Report
from .compare import compare_formats, ComparisonReport, higher_is_better
from .eval_performance import evaluate_performance, PerformanceReport
