"""
Analysis infrastructure for quantized operators.

Phase 4: full AnalysisContext + concrete Observers + AnalysisReport + Distribution taxonomy.
"""
from src.observer import QuantEvent, ObservableMixin, ObserverBase, SliceAwareObserver, SliceKey
from .observers import DistributionObserver, QSNRObserver, MSEObserver, HistogramObserver
from .context import AnalysisContext

from .report import AnalysisReport, Report  # Report is backward-compatible alias
from .compare import compare_formats, ComparisonReport, higher_is_better
from .eval_performance import evaluate_performance, PerformanceReport
from .e2e import Comparator, compare_models, compare_sessions
