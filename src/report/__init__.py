"""Output-driven reporting layer.

Post-hoc visualization is available via ``report.plot`` on ``StudyReport``.
Observer selection is driven by output key specs in ``_spec.py``.
"""

from src.report._session_report import SessionReport
from src.report._spec import resolve_outputs
from src.report._study_report import StudyReport

__all__ = [
    "SessionReport",
    "StudyReport",
    "resolve_outputs",
]
