"""
Analysis infrastructure for quantized operators.

Phase 4: full AnalysisContext + concrete Observers + AnalysisReport + Distribution taxonomy.
#
# Deprecation notice: ``Report`` (alias for ``AnalysisReport``) is deprecated.
# Prefer ``SessionReport`` and ``StudyReport`` from ``src.report`` for new code.
"""
from src.observer import QuantEvent, ObservableMixin, ObserverBase, SliceAwareObserver, SliceKey
from .observers import DistributionObserver, QSNRObserver, MSEObserver, HistogramObserver, DistributionFitObserver
from .context import AnalysisContext

from .report import AnalysisReport
from .report import Report  # Deprecated: use SessionReport / StudyReport from src.report instead
from .compare import compare_formats, ComparisonReport, higher_is_better
from .eval_performance import evaluate_performance, PerformanceReport
from .e2e import Comparator, compare_models, compare_sessions
from .correlation import DistributionProfile, DistributionTaxonomy, DistributionFitTaxonomy, TaxonomyAccessor
